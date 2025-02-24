include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
include("learning_player_base.jl")

function save_parameters_after_game_end(file::IO, board::Board, players::Vector{PlayerType}, player::EmpathRobotPlayer, winner_team::Symbol)
    features = compute_features(board, player.player)

    # For now, we just use a binary label to say who won
    label = get_csv_friendly(player.player.team == winner_team)
    values = join([get_csv_friendly(f[2]) for f in features], ",")
    
    println("values = $values,$label")
    write(file, "$values,$label\n")
end

# TODO implement this based on ML model, only accept trade if win proba augments more than the other player's win proba from the trade
# function choose_accept_trade(board::Board, player::RobotPlayer, from_player::PlayerPublicView, from_goods::Vector{Symbol}, to_goods::Vector{Symbol})::Bool
