include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
import Catan: choose_next_action, choose_place_robber

"""
    `get_action_with_features(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set{Symbol})`

Gets the legal action functions for the player at this board state, and computes the feature vector for each resulting state.
This is a critical helper function for all the machine-learning players.
"""
function get_action_with_features(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set{Symbol})
    action_functions = Catan.get_legal_action_functions(board, players, player, actions)
    reachable_states = []
    for (i,action_func!) in enumerate(action_functions)
        hypoth_board = deepcopy(board)
        hypoth_player = deepcopy(player)
        hypoth_game = Game([DefaultRobotPlayer(p.team) for p in players])
        action_func!(hypoth_game, hypoth_board, hypoth_player)
        features = compute_features(hypoth_board, hypoth_player.player)
        push!(reachable_states, (action_func!, features))
    end
    return reachable_states
end

"""
    `choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})`

Gathers all legal actions, and chooses the one that most increases the player's probability of victory, based on his `player.machine` model.  If no action increases the probability of victory, then do nothing.
"""
function choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})
    best_action_index = 0
    best_action_proba = -1
    #machine = ml_machine(player)
    machine = player.machine
    #current_features = compute_features(board, inner_player(player))
    current_features = compute_features(board, player.player)
    current_win_proba = predict_model(machine, current_features)
    # @info "$(inner_player(player).team) thinks his chance of winning is $(current_win_proba)"
    
    actions_and_features = get_action_with_features(board, players, player, actions)
    for (i,(action_func!, features)) in enumerate(actions_and_features)
        p = predict_model(machine, features)
        
        if p > best_action_proba
            best_action_proba = p
            best_action_index = i
        end
    end

    # Only do an action if it will improve his estimated chances of winning
    if best_action_proba > current_win_proba
        @info "And his chance of winning will go to $(best_action_proba) with this next move"
        return actions_and_features[best_action_index][1]
    end
    return nothing
end

function save_parameters_after_game_end(file::IO, board::Board, players::Vector{PlayerType}, player::PlayerType, winner_team::Symbol)
    features = compute_features(board, player.player)

    # For now, we just use a binary label to say who won
    label = get_csv_friendly(player.player.team == winner_team)
    values = join([get_csv_friendly(f[2]) for f in features], ",")
    write(file, "$values,$label\n")
end

# TODO implement this based on ML model, only accept trade if win proba augments more than the other player's win proba from the trade
# function choose_accept_trade(board::Board, player::RobotPlayer, from_player::PlayerPublicView, from_goods::Vector{Symbol}, to_goods::Vector{Symbol})::Bool
