
state_to_value = Dict()
function query_state_value(state_to_value, state_key, default = 0.5)
    if haskey(state_to_value, state_key)
        return state_to_value[state_key]
    else
        return default
    end
end
function query_state_value(process::MarkovRewardProcess, state_key, default = 0.5)
    if haskey(process.state_to_value, state_key)
        return process.state_to_value[state_key]
    elseif haskey(process.new_state_to_value, state_key)
        return process.new_state_to_value[state_key]
    else
        return default
    end
end

function update_state_value(process, state_key, new_value)
    @assert ~(haskey(process.state_to_value, state_key) && haskey(process.new_state_to_value, state_key))
    if haskey(process.state_to_value, state_key)
        process.state_to_value[state_key] = new_value
    else
        process.new_state_to_value[state_key] = new_value
    end
end


function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardMarkovPolicy, state)
    return get_combined_reward(process, policy.machine, state) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxValueMarkovPolicy, state)
    return query_state_value(process, state.key) 
end
function get_state_optimizing_quantity(process::MarkovRewardProcess, policy::MaxRewardPlusValueMarkovPolicy, state)
    return get_combined_reward(process, policy.machine, state) + query_state_value(process, state.key) 
end

function get_combined_reward(process::MarkovRewardProcess, machine::Machine, state)
    #value = query_state_value(state.key)
    model_proba = predict_model(machine, state.features)
    # TODO
    # win or loss feature is too difficult to calculate without passing game to feature computation
    # win_loss = state.features[:CountVictoryPoint]

    # Make sure return value is approximately on [0, 1] (technically vp can exceed 10)
    points = state.features[:CountVictoryPoint] / 10
    @assert process.model_coeff + process.points_coeff == 1.0
    reward = (process.model_coeff * model_proba) + (process.points_coeff * points)

    # Store in state for easy retrieval without re-computing inference from underlying model
    state.reward = reward
    return reward
end

"""
    `sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, current_state, reachable_states)`

Default implementation simply chooses the next state to maximize the `get_state_optimizing_quantity`.
"""
function sample_from_policy(process::MarkovRewardProcess, policy::MarkovPolicy, current_state, reachable_states)
    rewards = [get_state_optimizing_quantity(process, policy, s) for s in reachable_states]
    return argmax(rewards), reachable_states[argmax(rewards)]
end

"""
    `get_new_current_value(process, current_value, next_state, next_value)`

"""
function get_new_current_value(process, current_value, next_state, next_value)
    delta = next_state.reward + (process.reward_discount * next_value) - current_value
    return current_value + (process.learning_rate * delta)
end

"""
    `temporal_difference_step!(process::MarkovRewardProcess, policy::MarkovPolicy, current_state::MarkovState, state_to_value::Dict{UInt64, Float64}, reachable_states::Vector{MarkovState})`

Performs one step of tabular temporal difference (0)
"""
function temporal_difference_step!(process::MarkovRewardProcess, policy::MaxRewardMarkovPolicy, current_state::MarkovState, reachable_states::Vector{MarkovState})
    
    # sample from policy to get next state
    index, next_state = sample_from_policy(process, policy, current_state, reachable_states)
    
    # Update current state value
    current_value = query_state_value(process, current_state.key)
    next_value = query_state_value(process, next_state.key)
    new_current_value = get_new_current_value(process, current_value, next_state, next_value)
    update_state_value(process, current_state.key, new_current_value)
    return index, next_state, next_state.reward
end

"""
    `temporal_difference_step!(process::MarkovRewardProcess, policy::MarkovPolicy, current_state::MarkovState, state_to_value::Dict{UInt64, Float64}, reachable_states::Vector{MarkovState})`

Simply samples according to the policy, without updating states values.
"""
function temporal_difference_step!(process::MarkovRewardProcess, policy::MarkovPolicy, current_state::MarkovState, reachable_states::Vector{MarkovState})
    
    # sample from policy to get next state
    index, next_state = sample_from_policy(process, policy, current_state, reachable_states)
    next_state_quantity = get_state_optimizing_quantity(process, policy, next_state)
    return index, next_state, next_state_quantity
end
