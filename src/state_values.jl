function StateValueContainer(configs::Dict)
    master_state_to_value = read_values_file(configs["PlayerSettings"]["STATE_VALUES"])::Dict{UInt64, Float64}
    return StateValueContainer(master_state_to_value)
end

function StateValueContainer(master::Dict{UInt64, Float64})
    @info "Enriching MarkovPlayers with $(length(master)) pre-explored states"
    new_state_to_value = Dict{UInt64, Float64}()
    return StateValueContainer(master, new_state_to_value)
end

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


function write_values_file(values_file::String, state_values::AbstractVector{StateValueContainer})
    master = state_values[1].master
    currents = [s.current for s in state_values]
    write_values_file(values_file, master, currents)
end

function write_values_file(values_file::String, master_state_to_value::Dict{UInt64, Float64}, new_state_to_values::AbstractVector)
    merge!(master_state_to_value, new_state_to_values...)
    for s_to_v in new_state_to_values
        # and clear the new state to values learned
        empty!(s_to_v)
    end
    #println(master_state_to_value)
    write_values_file(values_file, master_state_to_value)
end

function write_values_file(values_file::String, state_to_value)
    data = join(["$k,$v\n" for (k,v) in collect(state_to_value)])
    file = open(values_file, "w")
    write(file, data)
    close(file)
end

function read_values_file(values_file::String, max_lines = nothing)::Dict{UInt64, Float64}
    if ~isfile(values_file)
        touch(values_file)
    end
    out = Dict{UInt64, Float64}() 
    #data = split(read(values_file, String), "\n")
    key_collisions = 0
    for (i,line) in enumerate(readlines(values_file))#, String))
        if !isnothing(max_lines) && i > max_lines
            break
        end
        if occursin(",", line)
            (key,value) = split(line, ",")
            parsed_key = parse(UInt64, key)
            if haskey(out, parsed_key)
                key_collisions += 1
                println("key collision: $key")
            end
            out[parse(UInt64, key)] = parse(Float64, value)
        end
    end
    if key_collisions > 0
        println("key collisions: $key_collisions")
    end
    return out
end