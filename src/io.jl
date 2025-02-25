function Catan.do_post_game_action(board::Board, players::Vector{PlayerType}, winner::LearningPlayer)
    return write_features_file(board::Board, players::Vector{PlayerType}, winner::PlayerType)
end
function Catan.do_post_game_action(board::Board, players::Vector{PlayerType}, winner::TemporalDifferencePlayer)
    return write_values_file(winner)
end

#
# Feature writing utils
#

function read_values_file(values_file::String)::Dict{UInt64, Float64}
    if ~isfile(values_file)
        io = open(values_file, "w")
        close(io)
    end
    out = Dict{UInt64, Float64}() 
    data = split(read(values_file, String), "\n")
    for line in data
        if occursin(",", line)
            (key,value) = split(line, ",")
            parsed_key = parse(UInt64, key)
            if haskey(out, parsed_key)
                println("key collision: $key")
            end
            out[parse(UInt64, key)] = parse(Float64, value)
        end
    end
    return out
end

function write_values_file(winner::TemporalDifferencePlayer)
    values_file = winner.io_config.values
    state_to_value = winner.process.new_state_to_value
    write_values_file(values_file, new_state_to_value)

    # Merge all new entries from this game into the main state_to_value dict
    merge!(winner.process.state_to_value, winner.process.new_state_to_value)
    # and clear the new state to values learned
    empty!(winner.process.new_state_to_value)
end

function write_values_file(values_file::String, state_to_value)
    data = join(["$k,$v\n" for (k,v) in collect(state_to_value)])
    file = open(values_file, "a")
    write(file, data)
    close(file)
end

function write_features_file(board::Board, players::Vector{PlayerType}, winner::PlayerType) 
    file = open(FEATURES_FILE, "a")
    _write_feature_file_header(file, board, winner)

    for player in players
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
get_csv_friendly(value::Bool) = string(Int(value))
get_csv_friendly(value::Symbol) = "\"$value\""
get_csv_friendly(value) = "\"$value\""
