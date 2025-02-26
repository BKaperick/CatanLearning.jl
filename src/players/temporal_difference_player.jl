include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
include("learning_player_base.jl")
include("../reinforcement.jl")

function choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::TemporalDifferencePlayer, actions::Set{Symbol})
    best_action_index = 0
    best_action_proba = -1
    machine = player.machine

    current_features = compute_features(board, player.player)
    current_state = MarkovState(current_features)
    current_quantity = get_state_optimizing_quantity(player.process, player.policy, current_state)
    actions_and_features = get_action_with_features(board, players, player, actions)
    reachable_states = [MarkovState(af[2]) for af in actions_and_features]
    
    if length(reachable_states) == 0
        return nothing
    end
    
    best_action_index, next_state, next_state_quantity = temporal_difference_step!(player.process, player.policy, current_state, reachable_states)
    
    # Only do an action if it will improve his optimized quantity
    if next_state_quantity > current_quantity
        #println("$next_state_quantity > $current_quantity")
        return actions_and_features[best_action_index][1]
    end

    return nothing
end
