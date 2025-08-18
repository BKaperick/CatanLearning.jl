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

struct TemporalDifferencePlayer <: MarkovPlayer
    player::Player
    model::DecisionModel
    model_public::DecisionModel
    process::MarkovRewardProcess
    policy::MarkovPolicy
    configs::Dict
    current_state::Union{Nothing, MarkovState}
end

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, configs::Dict)
    state_to_value = read_values_file(get_player_config(configs, "STATE_VALUES", team))
    return TemporalDifferencePlayer(TPolicy, team, state_to_value, Dict{UInt64, Float64}(), configs)
end
TemporalDifferencePlayer(team::Symbol, configs::Dict) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, configs)

TemporalDifferencePlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs::Dict) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs)

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs)
    model = try_load_serialized_model(team, configs)
    model_public = try_load_serialized_public_model(team, configs)
    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = TPolicy(model)
    TemporalDifferencePlayer(Player(team, configs), model, model_public, process, policy, configs, nothing)
end

struct HybridPlayer <: MarkovPlayer
    player::Player
    model::DecisionModel
    model_public::DecisionModel
    process::MarkovRewardProcess
    policy::MarkovPolicy
    configs::Dict
    current_state::Union{Nothing, MarkovState}
end

HybridPlayer(team::Symbol, configs::Dict) = HybridPlayer(team::Symbol, Dict{UInt64, Float64}(), Dict{UInt64, Float64}(), configs)
#HybridPlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs::Dict) 

function HybridPlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs)
    model = try_load_serialized_model(team, configs)::DecisionModel
    model_public = try_load_serialized_public_model(team, configs)

    reward_discount = get_player_config(configs, "REWARD_DISCOUNT", team)
    learning_rate = get_player_config(configs, "LEARNING_RATE", team)
    reward_weight = get_player_config(configs, "REWARD_WEIGHT", team)
    value_weight = get_player_config(configs, "VALUE_WEIGHT", team)
    process = MarkovRewardProcess(learning_rate, reward_discount, 1.0, 0.0, master_state_to_value, new_state_to_value)
    policy = WeightsRewardPlusValueMarkovPolicy(model, reward_weight, value_weight)
    player = Player(team, configs)
    HybridPlayer(player, model, model_public, process, policy, configs, nothing)
end

function Base.copy(player::TemporalDifferencePlayer)
    # Note, we `copy` only the player data, while the RL data should persist in order to pass updates the state info properly
    return TemporalDifferencePlayer(
        copy(player.player), 
        player.model, 
        player.model_public, 
        player.process, 
        player.policy, 
        player.configs,
        player.current_state
    )
end

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

function EmpathRobotPlayer(team::Symbol, configs::Dict) 
    EmpathRobotPlayer(
        Player(team, configs), 
        try_load_serialized_model(team, configs),
        try_load_serialized_public_model(team, configs)
    )
end

function Base.copy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(copy(player.player), player.model, player.model_public)
end

