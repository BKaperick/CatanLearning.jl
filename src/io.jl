function Catan.do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::EmpathRobotPlayer, winner::Union{PlayerType, Nothing})
    Catan.BoardApi.print_board(board)
end
function Catan.do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    write_public_features_file(game, board, players, player, winner)
    return write_main_features_file(game, board, players, player, winner)
end

function do_post_game_action(game::Game, board::Board, players::Vector{PlayerType}, player::TemporalDifferencePlayer, winner::Union{PlayerType, Nothing})
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
    values_file = player.io_config.values
    state_to_value = player.process.new_state_to_value
    write_values_file(values_file, state_to_value)

    # Merge all new entries from this game into the main state_to_value dict
    merge!(player.process.state_to_value, winner.process.new_state_to_value)
    # and clear the new state to values learned
    empty!(player.process.new_state_to_value)
    
    for player in players
        if hasproperty(player, :process)
            empty!(player.process.new_state_to_value)
            player.process.state_to_value = player.process.state_to_value
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
    features = compute_features_and_labels(game, board, player.player)
    _write_features_file(game, board, players, player, winner, file_name, features)
end

function write_public_features_file(game::Game, board::Board, players, player::PlayerType, winner::Union{PlayerType, Nothing}) 
    file_name = get_player_config(game.configs, "PUBLIC_FEATURES", player.player.team)
    features = compute_public_features_and_labels(game, board, player.player)
    _write_features_file(game, board, players, player, winner, file_name, features)
end

function _write_features_file(game::Game, board::Board, players, player::PlayerType, winner::Union{PlayerType, Nothing}, file_name::String, features::Vector) 
    file = open(file_name, "a")
    header = join([get_csv_friendly(f[1]) for f in features], ",")
    values = join([get_csv_friendly(f[2]) for f in features], ",")
    
    if filesize(file_name) == 0
        write(file, "$header\n")
    elseif game.turn_num == 1
        data, existing_header = readdlm(file_name, ',', header=true)
        @assert length(existing_header) == length(features) "Mismatch between existing feature schema in $file_name of length $(length(existing_header)) and current schema of length $(length(features))"
    end
    
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
