
state_to_value = Dict()

struct MarkovState
    # hash of game state to be used to track state value
    key::Int
    features::Dict{Symbol, Float}
    reward::Float
end
abstract type AbstractMarkovRewardProcess
end
struct MarkovRewardProcess <: AbstractMarkovRewardProcess
    reward_discount::Float
    learning_rate::Float
    win_loss_coeff::Float
    value_coeff::Float
    points_coeff::Float
end
abstract type MarkovPolicy
end

function query_state_value(state_key, default = 0.5)
    if haskey(state_to_value, state_key)
        return state_to_value[state_key]
    else
        return default
    end
end

function get_combined_reward(process::MarkovRewardProcess, state)
    value = query_state_value(state.key)
    # TODO
    # win or loss feature is too difficult to calculate without passing game to feature computation
    # win_loss = state.features[:CountVictoryPoint]

    # Make sure return value is approximately on [0, 1] (technically vp can exceed 10)
    points = state.features[:CountVictoryPoint] / 10
    @assert process.value_coeff + process.points_coeff == 1.0
    reward = (process.value_coeff * value) + (process.points_coeff * points)

    # Store in state for easy retrieval without re-computing inference from underlying model
    state.reward = reward
    return reward
end

"""
    `sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, current_state, reachable_states)`

Default implementation simply chooses the next state to maximize the estimated reward.
"""
function sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, current_state, reachable_states)
    rewards = [get_combined_reward(process, s) for s in reachable_states]
    return reachable_states[argmax(values)]
end

function temporal_difference(process::MarkovRewardProcess, policy::MarkovPolicy, init_state::MarkovState, state_to_value)
    for t=1:100
        state = temporal_difference(process, policy, state, state_to_value)
    end
end
function temporal_difference_step(process::MarkovRewardProcess, policy::MarkovPolicy, current_state::MarkovState, state_to_value)
    reachable_states = get_reachable_states(current_state)
    # sample from policy to get next state
    next_state = sample_from_policy(process, policy, current_state, reachable_states)
    
    # Update current state value
    current_value = query_state_value(current_state.key)
    next_value = query_state_value(next_state.key)
    delta = next_state.reward + (process.reward_discount * next_value) - current_value
    state_to_value[current_state.key] += process.learning_rate * delta
    return next_state
    #return state_to_value[current_state.key]
    #return temporal_difference(process, policy, next_state, state_to_value)
end
