"""
    query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64

Read from the state values cache to find the current state.  If not found, we retrieve the state's reward.
"""
function query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64
    @debug "querying key {$(state.key)} (searching $(length(keys(state_values.master))) + $(length(keys(state_values.current))) known values...)"
    if haskey(state_values.master, state.key)
        return state_values.master[state.key]
    elseif haskey(state_values.current, state.key)
        return state_values.current[state.key]
    else
        # Default to the state combined reward
        @debug "defaulting to reward = $(state.reward)"
        return state.reward
    end
end

function update_state_value(state_values::StateValueContainer, state_key::UInt, new_value::Float64)
    @assert ~(haskey(state_values.master, state_key) && haskey(state_values.current, state_key))
    if haskey(state_values.master, state_key)
        state_values.master[state_key] = new_value
    else
        state_values.current[state_key] = new_value
    end
end

function persistent_hash(features::Vector{Pair{Symbol, Float64}})
    # In order to avoid numerical instability issues in `Float64`, we apply rounding to the featurees first
    # Essentially applying a grid to our feature space, and considering all points the same if they are
    # within the same box.
    rounded_features = round.([f.second for f in features], digits=1)

    hash = UInt64(17)
    for f in rounded_features
        hash = hash * UInt64(23) + UInt64(f * 100)
    end
    return UInt64(hash)
end