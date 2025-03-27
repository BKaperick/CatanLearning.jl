"""
function Catan.do_post_game_action(board::Board, players::Vector{PlayerType}, winner::Nothing)
    println("in CL version 1")
    return
end
"""
function Catan.do_post_game_action(board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing})
    return write_features_file(board::Board, players, player, winner)
end

function do_post_game_action(board::Board, players::Vector{PlayerType}, player::TemporalDifferencePlayer, winner::Union{PlayerType, Nothing})
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

function write_features_file(board::Board, players, player::PlayerType, winner::Union{PlayerType, Nothing}) 
    file = open(FEATURES_FILE, "a")
    _write_feature_file_header(file, board, player)
    if winner == nothing
        save_parameters_after_game_end(file, board, players, player, nothing)
    else
        save_parameters_after_game_end(file, board, players, player, winner.player.team)
    end
    close(file)

end

function _write_feature_file_header(file::IO, board::Board, player::PlayerType)
    features = compute_features(board, player.player)
    header = join([get_csv_friendly(f[1]) for f in features], ",")
    label = get_csv_friendly("WonGame")
    if filesize(FEATURES_FILE) == 0
        write(file, "$header,$label\n")
    end
end

get_csv_friendly(value::Nothing) = "\"\""
get_csv_friendly(value::AbstractString) = "\"$value\""
get_csv_friendly(value::Int) = string(value)
get_csv_friendly(value::Int8) = string(value)
get_csv_friendly(value::AbstractFloat) = rstrip(rstrip(string(value), '0'),'.')
get_csv_friendly(value::Bool) = string(Int(value))
get_csv_friendly(value::Symbol) = "\"$value\""
get_csv_friendly(value) = "\"$value\""
