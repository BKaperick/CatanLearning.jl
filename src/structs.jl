using MLJ
import Catan: ChosenAction
import Base: show

struct Tournament
    games_per_map::Int
    maps_per_epoch::Int
    epochs::Int
    mutation_rule::Symbol
    unique_id::Int
end

function Tournament(configs::Dict, mutation_rule::Symbol) 
    @debug "$(configs["Tournament"]) $mutation_rule"
    id = rand(range(1,100_000))
    Tournament(configs["Tournament"]["GAMES_PER_MAP"], configs["Tournament"]["MAPS_PER_EPOCH"], configs["Tournament"]["NUM_EPOCHS"], mutation_rule, id)
end

function Tournament(configs::Dict) 
    @debug "$(configs["Tournament"]) $mutation_rule"
    id = rand(range(1,100_000))
    Tournament(configs["Tournament"]["GAMES_PER_MAP"], configs["Tournament"]["MAPS_PER_EPOCH"], configs["Tournament"]["NUM_EPOCHS"], :noop, id)
end

struct StateValueContainer
    master::Dict{UInt64, Float64}
    current::Dict{UInt64, Float64}
end

function Base.show(io::IO, s::StateValueContainer)
    print(io, "states: $(length(keys(s.master))) | $(length(keys(s.current)))")
end

function StateValueContainer(configs::Dict)
    master_state_to_value = read_values_file(configs["PlayerSettings"]["STATE_VALUES"])::Dict{UInt64, Float64}
    @info "Enriching MarkovPlayers with $(length(master_state_to_value)) pre-explored states"
    new_state_to_value = Dict{UInt64, Float64}()
    return StateValueContainer(master_state_to_value, new_state_to_value)
end

abstract type AbstractActionSet end
abstract type AbstractAction end
struct Action <: AbstractAction
    args::Tuple
    name::Symbol
    func!::Function
end

struct SampledAction <: AbstractAction
    args::Tuple
    name::Symbol
    # Deterministic version of `real_func!` that is used to calculate win proba of a deterministic branch
    func!::Function
    # This is the actual stochastic version of `func!` that is called during game play once this action is chosen
    #real_func!::Function
end

function Base.show(io::IO, a::AbstractAction)
    compact = get(io, :compact, false)
    if length(a.args) == 0
        print(io, "$(a.name)()")
        return
    end
    if compact
        print(io, "$(a.name)(...)")
    else
        print(io, "$(a.name)$(a.args)")
    end
end

function Base.show(io::IO, a::AbstractActionSet)
    compact = get(io, :compact, false)
    if length(a.actions) == 0
        print(io, "set $(a.name)()")
        return
    end
    if compact
        print(io, "$(a.name)($(length(a.actions)) actions)")
    else
        #  println(io, "$(a.name)($(length(a.actions)) actions)")
        print(io, "$(a.name)(\n$(join(["\ta: $a" for a in a.actions], "\n")))")
    end
end


mutable struct ActionSet{T<:AbstractAction} <: AbstractActionSet
    name::Symbol
    actions::Vector{T}
end

function Action(name::Symbol, func!::Function, args...) 
    Action(args, name, func!)
end
function SampledAction(name::Symbol, sampling_func!::Function, args...) 
    SampledAction(args, name, sampling_func!)
end
ActionSet(name::Symbol) = ActionSet(name, Vector{AbstractAction}([]))
function ActionSet{T}(name::Symbol) where {T<:AbstractAction}
    ActionSet(name, Vector{T}([])) 
end
function ActionSet(action::Action)
    return ActionSet{Action}(:Unnamed, [action]) 
end
#ActionSet{T}(name::Symbol) = ActionSet{T}(name, [], nothing)

struct MarkovState
    # hash of game state to be used to track state value
    key::UInt
    #features::Dict{Symbol, Float64}
    reward::Float64
end

abstract type AbstractMarkovRewardProcess
end

mutable struct MarkovRewardProcess <: AbstractMarkovRewardProcess
    reward_discount::AbstractFloat
    learning_rate::AbstractFloat
    # TODO not activated yet
    #win_loss_coeff::AbstractFloat
    
    # Coefficient for the ML model win probability term in combined reward function 
    model_coeff::AbstractFloat
    # Coefficient for the number of victory points term in combined reward function 
    points_coeff::AbstractFloat

    # Two dictionaries to keep track of state -> value mapping.
    # Do not access them directly! Use the following helper methods!
    # Writing: `update_state_value(process, state_key, new_value)`
    # Reading: `query_state_value(process, state, default = 0.5)`
    state_to_value::Dict{UInt64, Float64}
    new_state_to_value::Dict{UInt64, Float64}
end

abstract type DecisionModel
end
mutable struct MachineModel <: DecisionModel
    machine::Machine
end
mutable struct LinearModel <: DecisionModel
    weights::Vector{Float64}
end

"""
    `MarkovTransition`

Transitions are used as an intermediary data structure to allow TD-learning on non-deterministic actions, by calculating
reward/value as an average among the sampled possible actions.
"""
struct MarkovTransition
    #win_proba::Float64
    #victory_ponts::Int8
    states::Vector{MarkovState}
    chosen_action::ChosenAction
    reward::Float64
end

function Base.show(io::IO, t::MarkovTransition)
    compact = get(io, :compact, false)
    if compact
        print(io, "$(t.chosen_action) => $(t.reward)")
    else
        print(io, "$(length(t.states)) states | $(t.chosen_action) => $(t.reward)")
    end
end

#=
function MarkovTransition(process::MarkovRewardProcess, model::DecisionModel, action::AbstractAction)  
    states = [MarkovState(process, action.features, model)]
    return MarkoTransition(states, action)
end
=#

function MarkovTransition(states::Vector{MarkovState}, action::AbstractAction)
    reward = get_combined_reward(states)
    return MarkovTransition(states, ChosenAction(action.name, action.args...), reward)
end

abstract type MarkovPolicy end

"""
Samples reachable states by choosing maximum of reward + estimated value
"""
struct MaxRewardPlusValueMarkovPolicy <: MarkovPolicy
    # TODO do we need `model` field in `MarkovPolicy` ?
    model::DecisionModel
end

"""
Samples reachable states by choosing maximum of a * reward + b * estimated value
"""
struct WeightsRewardPlusValueMarkovPolicy <: MarkovPolicy
    model::DecisionModel
    reward_weight::Float64
    value_weight::Float64
end

"""
Samples reachable states by choosing maximum reward
"""
struct MaxRewardMarkovPolicy <: MarkovPolicy
    model::DecisionModel
end

"""
Samples reachable states by choosing maximum estimated value
"""
struct MaxValueMarkovPolicy <: MarkovPolicy
    model::DecisionModel
end

struct FeatureVector <: AbstractVector{Pair{Symbol, Float64}}
end

function MarkovState(process::MarkovRewardProcess, features::Vector{Pair{Symbol, Float64}}, model::DecisionModel)

    # TODO think about - should we also be rounding features for inference?
    # Currently mostly (only?) integer features, so we don't need to think too deeply yet
    reward = get_combined_reward(process, model, features)
    return MarkovState(features, reward)
end

"""
    MarkovState(features::Vector{Pair{Symbol, Float64}}, reward::Float64)

Represents one game state.  `Markov` because it maintains a key and reward,
which can be used in its modeling as a Markov Reward Process.
"""
function MarkovState(features::Vector{Pair{Symbol, Float64}}, reward::Float64)
    # In order to avoid numerical instability issues in `Float64`, we apply rounding to the featurees first
    # Essentially applying a grid to our feature space, and considering all points the same if they are
    # within the same box.
    rounded_features = round.([f.second for f in features], digits=1)

    return MarkovState(persistent_hash(rounded_features), reward)
end

function persistent_hash(feats)
    hash = UInt64(17)
    for f in feats
        hash = hash * UInt64(23) + UInt64(f * 100)
    end
    return UInt64(hash)
end


function get_combined_reward(states::Vector{MarkovState})
    # Get average reward from this transition
    reward = sum([s.reward for s in states]) / length(states)
    return reward
end