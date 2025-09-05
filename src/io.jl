using Catan: do_post_game_action, initialize_player

function Catan.do_post_game_action(game::Game, board::Board, players::AbstractVector{DefaultRobotPlayer}, winner::Union{PlayerType, Nothing})
    
    if game.configs["WRITE_FEATURES"] == true
        main_file_name = get_player_config(game.configs, "FEATURES", players[1].player.team)
        main_file = open(main_file_name, "a")

        public_file_name = get_player_config(game.configs, "PUBLIC_FEATURES", players[1].player.team)
        public_file = open(public_file_name, "a")
        
        for player in players
            features = compute_features_and_labels(game, board, player.player)
            _write_features_file(main_file, main_file_name, features)

            public_features = compute_public_features_and_labels(game, board, player.player)
            _write_features_file(public_file, public_file_name, public_features)
        end
    end
end

function Catan.initialize_player(board::Board, player::DefaultRobotPlayer)
end

function Catan.do_post_game_action(game::Game, board::Board, players::AbstractVector{PlayerType}, player::EmpathRobotPlayer, winner::Union{PlayerType, Nothing})
    if game.configs["PRINT_BOARD"]
        Catan.BoardApi.print_board(board)
    end
end

function Catan.do_post_game_action(game::Game, board::Board, players::AbstractVector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    if game.configs["WRITE_FEATURES"] == true
        write_public_features_file(game, board, players, player, winner)
        return write_main_features_file(game, board, players, player, winner)
    end
end

function Catan.do_post_game_produce!(channels::Dict{Symbol, Channel}, game::Game, board::Board, players::AbstractVector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    if game.configs["WRITE_FEATURES"] == true
        @debug "putting data"
        main_features = compute_features_and_labels(game, board, player.player)
        public_features = compute_public_features_and_labels(game, board, player.player)
        @debug "finished computing features"
        put!(channels[:main], main_features)
        @debug "Putting channel content to :main"
        put!(channels[:public], public_features)
        @debug "Putting channel content to :public"
    end
end

#=
function Catan.do_post_game_consume!(channels::Dict{Symbol, Channel}, output_ios::Dict{Symbol, IO}, game::Game, board::Board, players::AbstractVector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    _write_features_file
    main_features = take!(channels[:main])
    public_features = take!(channels[:public])
    _write_features_file(game, board, players, player, winner, output_ios[:main], game.configs["FEATURES"], main_features)
    _write_features_file(game, board, players, player, winner, output_ios[:public], game.configs["PUBLIC_FEATURES"], public_features)
end
=#
function do_post_game_consume!(channels::Dict{Symbol, Channel}, game::Game, board::Board, players::AbstractVector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    do_post_game_consume!(channels, game.configs)
end

function do_post_game_consume!(channels::Dict{Symbol, Channel}, configs::Dict)
    println(channels)
    while ~isempty(channels[:main])
        consume_channel!(channels[:main], configs["PlayerSettings"]["FEATURES"])
    end
    while ~isempty(channels[:public])
        consume_channel!(channels[:public], configs["PlayerSettings"]["PUBLIC_FEATURES"])
    end
end

function consume_channel!(channel, file_name)
    features = take!(channel)
    @debug "Consuming channel content to $file_name !"
    _write_features_file(file_name, features)
end

function Catan.do_post_game_action(game::Game, board::Board, players::AbstractVector{PlayerType}, player::MarkovPlayer, winner::Union{PlayerType, Nothing})
    if game.configs["WRITE_FEATURES"] == true
        write_public_features_file(game, board, players, player, winner)
        write_main_features_file(game, board, players, player, winner)
    end
    if game.configs["WRITE_VALUES"] == true
        first_markov = [p.player.team for p in players if p isa MarkovPlayer][1]
        if player.player.team == first_markov
            write_values_file(players, player)
        end
    end
end

#
# Feature writing utils
#

function write_values_file(players::AbstractVector{PlayerType}, player::MarkovPlayer)
    values_file = get_player_config(player, "STATE_VALUES")
    @info "Writing values to $values_file"

    # Merge all new entries from this game into the main state_to_value dict
    state_values = [p.process.state_values for p in players if p isa MarkovPlayer]
    write_values_file(values_file, state_values)
end

function write_main_features_file(game::Game, board::Board, players, player::PlayerType, winner::Union{PlayerType, Nothing}) 
    file_name = get_player_config(game.configs, "FEATURES", player.player.team)
    file = open(file_name, "a")
    features = compute_features_and_labels(game, board, player.player)
    _write_features_file(file, file_name, features)
end

function write_public_features_file(game::Game, board::Board, players, player::PlayerType, winner::Union{PlayerType, Nothing}) 
    file_name = get_player_config(game.configs, "PUBLIC_FEATURES", player.player.team)
    file = open(file_name, "a")
    features = compute_public_features_and_labels(game, board, player.player)
    _write_features_file(file, file_name, features)
end

function write_features_header_if_needed(path, features, configs)
    columns = vcat(features, get_labels())

    if ~isfile(path)
        touch(path)
    end

    open(path, "a") do file
        header = join(get_csv_friendly.(first.(columns)), ",")
        if filesize(path) == 0
            write(file, "$header\n")
        else
            existing_header_size = size(CSV.read(path, DataFrame, limit=0))[2]
            @assert existing_header_size == length(columns) "Mismatch between existing feature schema in $path of length $(existing_header_size) and current schema of length $(length(columns))"
        end
    end
end


function _write_features_file(file_name, features::Vector) 
    _write_features_file(open(file_name, "a"), file_name, features)
end
function _write_many_features_file(channel, count, file_name) 
    file = open(file_name, "a")
    for i=1:count
        feats = take!(channel)
        values = join([get_csv_friendly(f[2]) for f in feats], ",")
        write(file, "$values\n")
    end
    close(file)
end

function _write_features_file(file::IO, file_name, features::Vector) 
    values = join([get_csv_friendly(f[2]) for f in features], ",")
    
    write(file, "$values\n")
    close(file)
end

get_csv_friendly(value::Nothing) = "\"\""
get_csv_friendly(value::AbstractString) = "\"$value\""
get_csv_friendly(value::Int) = string(value)
get_csv_friendly(value::Int8) = string(value)
get_csv_friendly(value::Int32) = string(value)
get_csv_friendly(value::AbstractFloat) = rstrip(rstrip(string(value), '0'),'.')
get_csv_friendly(value::Bool) = string(Int(value))
get_csv_friendly(value::Symbol) = "\"$value\""
get_csv_friendly(value) = "\"$value\""
