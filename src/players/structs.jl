using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: hash
using DelimitedFiles

abstract type LearningPlayer <: RobotPlayer
end

abstract type MarkovPlayer <: LearningPlayer
end

struct EmpathRobotPlayer <: LearningPlayer 
    player::Player
    model::DecisionModel
    model_public::DecisionModel
end

mutable struct HybridPlayer <: MarkovPlayer
    const player::Player
    model::DecisionModel
    model_public::DecisionModel
    const process::MarkovRewardProcess
    const policy::MarkovPolicy
    const configs::Dict
    current_state::Union{Nothing, MarkovState}
end

function HybridPlayer(player::Player, team::Symbol, configs::Dict)
    svc = StateValueContainer(configs)
    return HybridPlayer(player, svc, team, configs)
end
function HybridPlayer(player::Player, svc::StateValueContainer, team::Symbol, configs::Dict)
    model = try_load_serialized_model(team, configs)::DecisionModel
    model_public = try_load_serialized_public_model(team, configs)

    reward_discount = get_player_config(configs, "REWARD_DISCOUNT", team)
    learning_rate = get_player_config(configs, "LEARNING_RATE", team)
    reward_weight = get_player_config(configs, "REWARD_WEIGHT", team)
    value_weight = get_player_config(configs, "VALUE_WEIGHT", team)
    process = MarkovRewardProcess(learning_rate, reward_discount, 1.0, 0.0, svc)
    policy = WeightsRewardPlusValueMarkovPolicy(model, reward_weight, value_weight)
    HybridPlayer(player, model, model_public, process, policy, configs, nothing)
end
HybridPlayer(team::Symbol, configs::Dict) = HybridPlayer(Player(team, configs), team, configs)
HybridPlayer(player::Player) = HybridPlayer(player, player.team, player.configs)
HybridPlayer(player::Player, svc::StateValueContainer) = HybridPlayer(player, svc, player.team, player.configs)

function Base.copy(player::HybridPlayer)
    # Note, we `copy` only the player data, while the RL data should persist in order to pass updates the state info properly
    return HybridPlayer(
        copy(player.player), 
        player.model, 
        player.model_public, 
        player.process, 
        player.policy, 
        player.configs,
        player.current_state
    )
end

function EmpathRobotPlayer(player::Player, team::Symbol, configs::Dict) 
    EmpathRobotPlayer(
        player, 
        try_load_serialized_model(team, configs),
        try_load_serialized_public_model(team, configs)
    )
end
EmpathRobotPlayer(team::Symbol, configs::Dict) = EmpathRobotPlayer(Player(team, configs), team, configs)
EmpathRobotPlayer(player::Player) = EmpathRobotPlayer(player, player.team, player.configs)

function Base.copy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(copy(player.player), player.model, player.model_public)
end

