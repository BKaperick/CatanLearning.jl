using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: deepcopy,hash
using DelimitedFiles

abstract type LearningPlayer <: RobotPlayer
end

mutable struct EmpathRobotPlayer <: LearningPlayer 
    player::Player
    machine::Machine
    machine_public::Machine
end

mutable struct MutatedEmpathRobotPlayer <: LearningPlayer 
    player::Player
    machine::Machine
    machine_public::Machine
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

mutable struct TemporalDifferencePlayer <: LearningPlayer
    player::Player
    machine::Machine
    machine_public::Machine
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
    machine = try_load_model_from_csv(team, configs)
    machine_public = try_load_public_model_from_csv(team, configs)

    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = TPolicy(machine)
    TemporalDifferencePlayer(Player(team, configs), machine, machine_public, process, policy, configs, nothing)
end

mutable struct HybridPlayer <: LearningPlayer
    player::Player
    decision_model_weights::Array{Float64}
    machine_public::Machine
    process::MarkovRewardProcess
    policy::MarkovPolicy
    configs::Dict
    current_state::Union{Nothing, MarkovState}
end

HybridPlayer(team::Symbol, configs::Dict) = HybridPlayer(team::Symbol, configs)
HybridPlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs::Dict) = HybridPlayer(WeightsRewardPlusValueMarkovPolicy, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs)

function HybridPlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, configs)
    model_weights = try_load_linear_model_from_csv(team, configs)
    machine_public = try_load_public_model_from_csv(team, configs)

    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    reward_weight = get_player_config(configs, "REWARD_WEIGHT", team)
    value_weight = get_player_config(configs, "VALUE_WEIGHT", team)
    policy = WeightsRewardPlusValueMarkovPolicy(nothing, reward_weight, value_weight)
    HybridPlayer(Player(team, configs), model_weights, machine_public, process, policy, configs, nothing)
end

function Base.deepcopy(player::MutatedEmpathRobotPlayer)
    return MutatedEmpathRobotPlayer(deepcopy(player.player), player.machine, player.machine_public, deepcopy(player.mutation), player.configs) 
end

function Base.deepcopy(player::TemporalDifferencePlayer)
    # Note, we deepcopy only the player data, while the RL data should persist in order to pass updates the state info properly
    return TemporalDifferencePlayer(
        deepcopy(player.player), 
        player.machine, 
        player.machine_public, 
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
        player.decision_model_weights, 
        player.machine_public, 
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
    return EmpathRobotPlayer(deepcopy(player.player), player.machine, player.machine_public)
end

