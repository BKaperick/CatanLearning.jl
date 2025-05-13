using Catan: do_post_game_action, initialize_player

function Catan.do_post_game_action(game::Game, board::Board, players::Vector{DefaultRobotPlayer}, winner::Union{PlayerType, Nothing})
    
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
    if board.configs["WRITE_FEATURES"] == true
        @info "Intializing player feature files"
        write_features_header_if_needed(get_player_config(player, "FEATURES"), get_features())
        write_features_header_if_needed(get_player_config(player, "PUBLIC_FEATURES"), get_public_features())
    end
end

function Catan.do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::EmpathRobotPlayer, winner::Union{PlayerType, Nothing})
    if game.configs["PRINT_BOARD"]
        Catan.BoardApi.print_board(board)
    end
end
function Catan.do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    write_public_features_file(game, board, players, player, winner)
    return write_main_features_file(game, board, players, player, winner)
end

function Catan.do_post_game_produce!(channels::Dict{Symbol, Channel}, game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    main_features = compute_features_and_labels(game, board, player.player)
    public_features = compute_public_features_and_labels(game, board, player.player)
    put!(channels[:main], main_features)
    toggleprint("Putting channel content to :main")
    put!(channels[:public], public_features)
    toggleprint("Putting channel content to :public")
end

#=
function Catan.do_post_game_consume!(channels::Dict{Symbol, Channel}, output_ios::Dict{Symbol, IO}, game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    _write_features_file
    main_features = take!(channels[:main])
    public_features = take!(channels[:public])
    _write_features_file(game, board, players, player, winner, output_ios[:main], game.configs["FEATURES"], main_features)
    _write_features_file(game, board, players, player, winner, output_ios[:public], game.configs["PUBLIC_FEATURES"], public_features)
end
=#
function do_post_game_consume!(channels::Dict{Symbol, Channel}, game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
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
    toggleprint("Consuming channel content to $file_name !")
    _write_features_file(file_name, features)
end

function Catan.do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::TemporalDifferencePlayer, winner::Union{PlayerType, Nothing})
    println("writing values")
    return write_values_file(players, player)
end

#
# Feature writing utils
#

function read_values_file(values_file::String, max_lines = nothing)::Dict{UInt64, Float64}
    println("reading values file...")
    if ~isfile(values_file)
        io = open(values_file, "w")
        close(io)
    end
    out = Dict{UInt64, Float64}() 
    #data = split(read(values_file, String), "\n")
    key_collisions = 0
    for (i,line) in enumerate(readlines(values_file))#, String))
        if line == max_lines
            break
        end
        if occursin(",", line)
            (key,value) = split(line, ",")
            parsed_key = parse(UInt64, key)
            if haskey(out, parsed_key)
                key_collisions += 1
                #println("key collision: $key")
            end
            out[parse(UInt64, key)] = parse(Float64, value)
        end
    end
    if key_collisions > 0
        println("key collisions: $key_collisions")
    end
    return out
end

function write_values_file(players::Vector{PlayerType}, player::TemporalDifferencePlayer)
    values_file = get_player_config(player, "STATE_VALUES")
    state_to_value = player.process.new_state_to_value
    write_values_file(values_file, state_to_value)

    # Merge all new entries from this game into the main state_to_value dict
    merge!(player.process.state_to_value, player.process.new_state_to_value)
    # and clear the new state to values learned
    empty!(player.process.new_state_to_value)
    
    for other_player in players
        if hasproperty(other_player, :process)
            empty!(other_player.process.new_state_to_value)
            other_player.process.state_to_value = player.process.state_to_value
        end
    end
end

function write_values_file(values_file::String, state_to_value)
    data = join(["$k,$v\n" for (k,v) in collect(state_to_value)])
    file = open(values_file, "a")
    write(file, data)
    close(file)
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



function write_features_header_if_needed(path, features)
    columns = vcat(features, get_labels())

    if ~isfile(path)
        touch(path)
    end

    open(path, "a") do file
        header = join(get_csv_friendly.(first.(columns)), ",")
        if filesize(path) == 0
            write(file, "$header\n")
        else
            data, existing_header = readdlm(path, ',', header=true)
            @assert length(existing_header) == length(columns) "Mismatch between existing feature schema in $path of length $(length(existing_header)) and current schema of length $(length(columns))"
        end
    end
end

function _write_features_file(file_name, features::Vector) 
    _write_features_file(open(file_name, "a"), file_name, features)
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
