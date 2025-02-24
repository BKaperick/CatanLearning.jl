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

function MarkovState(features)
    return MarkovState(hash(features), features, nothing)
end

abstract type AbstractMarkovRewardProcess
end
struct MarkovRewardProcess <: AbstractMarkovRewardProcess
    reward_discount::AbstractFloat
    learning_rate::AbstractFloat
    win_loss_coeff::AbstractFloat
    value_coeff::AbstractFloat
    points_coeff::AbstractFloat
end
abstract type MarkovPolicy
end

