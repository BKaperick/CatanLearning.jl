using MLJ

struct Tournament
    games_per_map::Int
    maps_per_epoch::Int
    epochs::Int
    mutation_rule::Symbol
end

struct MarkovState
    # hash of game state to be used to track state value
    key::UInt
    features::Dict{Symbol, AbstractFloat}
    reward::Union{Nothing, AbstractFloat}
end

struct MarkovPolicy
    machine::Machine
end

function MarkovState(features)
    return MarkovState(hash(features), features, nothing)
end

abstract type AbstractMarkovRewardProcess
end
struct MarkovRewardProcess <: AbstractMarkovRewardProcess
    reward_discount::AbstractFloat
    learning_rate::AbstractFloat
    # TODO not activated yet
    #win_loss_coeff::AbstractFloat
    
    # Coefficient for the ML model win probability term in combined reward function 
    model_coeff::AbstractFloat
    # Coefficient for the number of victory points term in combined reward function 
    points_coeff::AbstractFloat
end
