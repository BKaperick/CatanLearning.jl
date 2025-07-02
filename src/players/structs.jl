using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: deepcopy,hash
using DelimitedFiles

abstract type LearningPlayer <: RobotPlayer
end

abstract type MarkovPlayer <: LearningPlayer
end

mutable struct EmpathRobotPlayer <: LearningPlayer 
    player::Player
    model::DecisionModel
    model_public::DecisionModel
end

mutable struct MutatedEmpathRobotPlayer <: LearningPlayer 
    player::Player
    model::DecisionModel
    model_public::DecisionModel
    mutation::Dict #{Symbol, AbstractFloat}
    configs::Dict
end

MutatedEmpathRobotPlayer(team::Symbol, configs::Dict) = MutatedEmpathRobotPlayer(
    team, Dict{Symbol, AbstractFloat}(), configs)

function MutatedEmpathRobotPlayer(team::Symbol, mutation::Dict, configs::Dict) 
    MutatedEmpathRobotPlayer(
    Player(team, configs), 
    try_load_model_from_csv(team, configs),
    try_load_public_model_from_csv(team, configs),
    mutation,
    configs)
end

mutable struct TemporalDifferencePlayer <: MarkovPlayer
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
    model = try_load_model_from_csv(team, configs)
    model_public = try_load_public_model_from_csv(team, configs)
    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = TPolicy(model)
    TemporalDifferencePlayer(Player(team, configs), model, model_public, process, policy, configs, nothing)
end

mutable struct HybridPlayer <: MarkovPlayer
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
    model = try_load_linear_model_from_csv(team, configs)::LinearModel
    model_public = try_load_public_model_from_csv(team, configs)

    reward_discount = get_player_config(configs, "REWARD_DISCOUNT", team)
    learning_rate = get_player_config(configs, "LEARNING_RATE", team)
    reward_weight = get_player_config(configs, "REWARD_WEIGHT", team)
    value_weight = get_player_config(configs, "VALUE_WEIGHT", team)
    process = MarkovRewardProcess(learning_rate, reward_discount, 1.0, 0.0, master_state_to_value, new_state_to_value)
    policy = WeightsRewardPlusValueMarkovPolicy(model, reward_weight, value_weight)
    player = Player(team, configs)
    HybridPlayer(player, model, model_public, process, policy, configs, nothing)
end

function Base.deepcopy(player::MutatedEmpathRobotPlayer)
    return MutatedEmpathRobotPlayer(deepcopy(player.player), player.model, player.model_public, deepcopy(player.mutation), player.configs) 
end

function Base.deepcopy(player::TemporalDifferencePlayer)
    # Note, we deepcopy only the player data, while the RL data should persist in order to pass updates the state info properly
    return TemporalDifferencePlayer(
        deepcopy(player.player), 
        player.model, 
        player.model_public, 
        player.process, 
        player.policy, 
        player.configs,
        deepcopy(player.current_state)
    )
end

function Base.deepcopy(player::HybridPlayer)
    # Note, we deepcopy only the player data, while the RL data should persist in order to pass updates the state info properly
    return HybridPlayer(
        deepcopy(player.player), 
        player.model, 
        player.model_public, 
        player.process, 
        player.policy, 
        player.configs,
        deepcopy(player.current_state)
    )
end

function EmpathRobotPlayer(team::Symbol, configs::Dict) 
    EmpathRobotPlayer(
        Player(team, configs), 
        try_load_model_from_csv(team, configs),
        try_load_public_model_from_csv(team, configs)
    )
end

function Base.deepcopy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(deepcopy(player.player), player.model, player.model_public)
end

