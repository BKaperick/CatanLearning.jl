function query_state_value(process::MarkovRewardProcess, transition::MarkovTransition)
    # Get average value from this transition
    value = sum([query_state_value(process, s) for s in transition.states]) / length(transition.states)
    return value
end

function query_state_value(process::MarkovRewardProcess, state::MarkovState)
    @debug "querying key {$(state.key)} (searching $(length(keys(process.state_to_value))) + $(length(keys(process.new_state_to_value))) known values...)"
    if haskey(process.state_to_value, state.key)
        return process.state_to_value[state.key]
    elseif haskey(process.new_state_to_value, state.key)
        return process.new_state_to_value[state.key]
    else
        # Default to the state combined reward
        @debug "defaulting to reward = $(state.reward)"
        return state.reward
    end
end

function update_state_value(process::MarkovRewardProcess, state_key::UInt, new_value::Float64)
    @assert ~(haskey(process.state_to_value, state_key) && haskey(process.new_state_to_value, state_key))
    if haskey(process.state_to_value, state_key)
        process.state_to_value[state_key] = new_value
    else
        process.new_state_to_value[state_key] = new_value
    end
end

function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardMarkovPolicy, state::MarkovState)
    return state.reward
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardMarkovPolicy, transition::MarkovTransition)
    return transition.reward
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxValueMarkovPolicy, state::MarkovState)
    return query_state_value(process, state) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxValueMarkovPolicy, transition::MarkovTransition)
    return query_state_value(process, transition) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardPlusValueMarkovPolicy, state::MarkovState)
    return state.reward + query_state_value(process, state) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardPlusValueMarkovPolicy, transition::MarkovTransition)
    return transition.reward + query_state_value(process, transition) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::WeightsRewardPlusValueMarkovPolicy, transition::MarkovTransition)
    reward = policy.reward_weight * transition.reward
    value = policy.value_weight * query_state_value(process, transition)
    return reward + value
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::WeightsRewardPlusValueMarkovPolicy, state::MarkovState)
    reward = policy.reward_weight * state.reward
    value = policy.value_weight * query_state_value(process, state)
    println("$(state.key): $(reward + value)")
    return reward + value
end

"""
    `sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, reachable_states)`

Default implementation simply chooses the next state to maximize the `get_state_optimizing_quantity`.
"""
function sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, transitions::Vector{MarkovTransition})
    rewards = [get_state_optimizing_quantity(process, policy, t) for t in transitions]
    for r in rewards
        @assert r !== nothing
    end
    return maximum(rewards), argmax(rewards), transitions[argmax(rewards)]
end

"""
    `get_new_current_value(process, current_value, next_state, next_value)`

"""
function get_new_current_value(process, current_value, next_state, next_value)
    next_discounted = (process.reward_discount * next_value)
    step = next_state.reward + next_discounted
    delta = step - current_value
    return current_value + (process.learning_rate * delta)
end

function finish_temporal_difference_step!(process::MarkovRewardProcess, 
        current_state::MarkovState, next_state::MarkovState)
    @assert next_state.reward !== nothing
    
    # Update current state value
    current_value = query_state_value(process, current_state)

    next_value = query_state_value(process, next_state)
    new_current_value = get_new_current_value(process, current_value, next_state, 
                                              next_value)
    update_state_value(process, current_state.key, new_current_value)
    return next_state, next_state.reward
end