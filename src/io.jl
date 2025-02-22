function do_post_game_action(board::Board, players::Vector{PlayerType}, winner::PlayerType)
    return write_features_file(board::Board, players::Vector{PlayerType}, winner::PlayerType)
end

#
# Feature writing utils
#

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
