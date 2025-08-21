include("../reinforcement.jl")
using Catan: do_post_action_step, choose_next_action, get_state_score

function Catan.do_post_action_step(board::Board, player::MarkovPlayer)
    # We can't re-use already computed features, because if it was a non-deterministic action, the features may not be the same
    next_features = compute_features(board, player.player)
    next_state = MarkovState(player.process, next_features, player.model)

    @assert next_state.reward !== nothing
    finish_temporal_difference_step!(player.process, player.current_state, next_state::MarkovState)
end


function Catan.get_state_score(board::Board, player::MarkovPlayer)::Float64
    features = compute_features(board, player.player)
    reward = get_combined_reward(player.player.process, player.player.model, features)
    return reward
end

function get_combined_reward(process::MarkovRewardProcess, model::DecisionModel, features::Vector{Pair{Symbol, Float64}})::Float64
    model_proba = predict_model(model, features)
    
    # TODO
    # win or loss feature is too difficult to calculate without passing game to feature computation
    # win_loss = state.features[:CountVictoryPoint]

    # Make sure return value is approximately on [0, 1] (technically vp can exceed 10)
    points = Dict(features)[:CountVictoryPoint] / 10
    @assert process.model_coeff + process.points_coeff == 1.0
    reward = (process.model_coeff * model_proba) + (process.points_coeff * points)
    return reward
end