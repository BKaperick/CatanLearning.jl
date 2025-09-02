using MLJ
import Catan: ChosenAction
import Base: show

abstract type AbstractTournament
end

generate_tournament_id()::Int = rand(range(1,100_000))

struct TournamentConfig
    games_per_map::Int
    maps_per_epoch::Int
    epochs::Int
    generate_random_maps::Bool
    unique_id::Int
    path::String
end

struct Tournament <: AbstractTournament
    configs::TournamentConfig
    teams::AbstractVector{Symbol}
    winners::Dict{Union{Symbol, Nothing}, Int}
    #Tournament(configs::Dict) = Tournament(TournamentConfig(configs["Tournament"]))
end

struct MutatingTournament <: AbstractTournament
    configs::TournamentConfig
    teams::AbstractVector{Symbol}
    winners::Dict{Union{Symbol, Nothing}, Int}
    mutation_rule::Symbol
    #MutatingTournament(configs::Dict) = MutatingTournament(TournamentConfig(configs["Tournament"]))
end

struct AsyncTournament <: AbstractTournament
    configs::TournamentConfig
    teams::AbstractVector{Symbol}
    channels::Dict{Symbol, Channel}
    #AsyncTournament(configs::Dict) = AsyncTournament(TournamentConfig(configs["Tournament"]),
    #channels = Catan.read_channels_from_config(configs))
end

struct StateValueContainer
    master::LMDBDict{UInt64, Float64}
    env::Environment
    path::AbstractString
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


struct ActionSet{T<:AbstractAction} <: AbstractActionSet
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

    # Container for two dictionaries keeping track of state -> value mapping.
    # Do not access it directly! Use the following helper methods!
    # Writing: `update_state_value(process, state_key, new_value)`
    # Reading: `query_state_value(process, state, default = 0.5)`
    state_values::StateValueContainer
end

function MarkovRewardProcess(r::AbstractFloat, l::AbstractFloat, m::AbstractFloat, p::AbstractFloat, master::LMDBDict{UInt64, Float64}, current::Dict{UInt64, Float64})
    return MarkovRewardProcess(r, l, m, p, StateValueContainer(master, current))
end

abstract type DecisionModel
end
mutable struct MachineModel <: DecisionModel
    machine::Machine
end
mutable struct LinearModel <: DecisionModel
    weights::Vector{Float64}
end
mutable struct EmptyModel <: DecisionModel
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
    return MarkovState(persistent_hash(features), reward)
end


function get_combined_reward(states::Vector{MarkovState})
    # Get average reward from this transition
    reward = sum([s.reward for s in states]) / length(states)
    return reward
end