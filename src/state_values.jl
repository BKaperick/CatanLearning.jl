using LMDB


StateValueContainer(configs::Dict{String, Any}) = StateValueContainer(configs["PlayerSettings"]["STATE_VALUES"])

function StateValueContainer(path::AbstractString)
    if ~ispath(path)
        mkpath(path)
    end
    master = LMDBDict{UInt64, Float64}(path)
    master.env[:MapSize] = 4_000_000_000
    return StateValueContainer(master, master.env, path)
end

"""
    query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64

Read from the state values cache to find the current state.  If not found, we retrieve the state's reward.
"""
function query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64
    get(state_values.master, state.key, state.reward)
end

function update_state_value(state_values::StateValueContainer, state_key::UInt64, new_value::Float64)
    state_values.master[state_key] = new_value
end

function update_state_values(state_values::StateValueContainer, new_values::AbstractVector{Dict{UInt, Float64}})
    for (k,v) in merge(new_values...)
        update_state_value(state_values, k, v)
    end
end

function write_values_file(values_file::String, state_values::AbstractVector{StateValueContainer})
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
