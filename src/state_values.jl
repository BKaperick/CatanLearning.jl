using LMDB


StateValueContainer(configs::Dict{String, Any}) = StateValueContainer(configs["PlayerSettings"]["STATE_VALUES"])

function StateValueContainer(path::AbstractString)
    if ~ispath(path)
        mkpath(path)
    end

    env = create()
    #env[:MapSize] = 1000^3
    #open(env, path)
    key_path = joinpath(path, "_bryan_keys")
    if isfile(key_path)
        keys = Set([parse(i,UInt64) for i in readlines(open(key_path, "r"))])
    else
        keys = Set{UInt64}()
    end

    return StateValueContainer(env, path, keys, key_path)
end

"""
    query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64

Read from the state values cache to find the current state.  If not found, we retrieve the state's reward.
"""
function query_state_value(state_values::StateValueContainer, state::MarkovState)::Float64
    if !(state.key in state_values.keys)
        return state.reward
    end
    env = state_values.env
    try
        open(env, state_values.path) # open db environment !!! `testdb` must exist !!!
        txn = start(env)      # start new transaction
        dbi = open(txn)       # open database
        try
            x = get(txn, dbi, state.key, Float64) # add key-value pair
            @info "found $(state.key) -> $x"
            #=
            catch LMDBError
                println("hi")
                # Default to the state combined reward
                @debug "defaulting to reward = $(state.reward)"
                return state.reward
            end
            =#
            commit(txn)                  # commit transaction
        finally
            close(env, dbi)  # close db
        end
    finally
        close(env)           # close environment
    end

    return x


    
end

function update_state_value(state_values::StateValueContainer, state_key::UInt, new_value::Float64)
    push!(state_values.keys, state_key)
    env = state_values.env
    try
        open(env, state_values.path) # open db environment !!! `testdb` must exist !!!
        txn = start(env)      # start new transaction
        dbi = open(txn)       # open database
        try
            x = put!(txn, dbi, state_key, new_value) # add key-value pair
            commit(txn)                  # commit transaction
        finally
            close(env, dbi)  # close db
        end
    finally
        close(env)           # close environment
    end
end
function update_state_values(state_values::StateValueContainer, new_values::AbstractVector{Dict{UInt, Float64}})
    for (k,v) in merge(new_values...)
        update_state_value(state_values, k, v)
    end
end

function write_values_file(values_file::String, state_values::AbstractVector{StateValueContainer})
    merge!(state_values[1].keys, [s.keys for s in state_values[2:end]])
    io = open(state_values[1].key_path, "w")
    for k in state_values[1].keys
        println(io, k)
    end
    close(io)
    #=
    merge!(state_values[1].current, [s.current for s in state_values[2:end]]...)
    update_state_values(state_values[1], state_values[1].current)
    for (k,v) in state_values[1].current
        update_state_value(state_values[1], k, v)
    end
    =#
end

function _deprecated_read_values_file(values_file::String, max_lines = nothing)::Dict{UInt64, Float64}
    if ~isfile(values_file)
        touch(values_file)
    end
    out = Dict{UInt64, Float64}() 
    key_collisions = 0
    for (i,line) in enumerate(readlines(values_file))
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
