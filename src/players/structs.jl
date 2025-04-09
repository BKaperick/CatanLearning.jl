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
    machine_public::Union{Nothing, Machine}
end

mutable struct MutatedEmpathRobotPlayer <: LearningPlayer 
    player::Player
    machine::Machine
    machine_public::Union{Nothing, Machine}
    mutation::Dict #{Symbol, AbstractFloat}
    io_config::IoConfig
end

MutatedEmpathRobotPlayer(team::Symbol, player_configs::Dict) = MutatedEmpathRobotPlayer(
    team, Dict{Symbol, AbstractFloat}(), player_configs)

MutatedEmpathRobotPlayer(team::Symbol, mutation::Dict, player_configs::Dict) = MutatedEmpathRobotPlayer(team, mutation, IoConfig(player_configs))

function MutatedEmpathRobotPlayer(team::Symbol, mutation::Dict, io_config::IoConfig) 
    MutatedEmpathRobotPlayer(
    Player(team), 
    try_load_model_from_csv(io_config),
    nothing,
    mutation,
    io_config)
end

mutable struct TemporalDifferencePlayer <: LearningPlayer
    player::Player
    machine::Machine
    machine_public::Union{Nothing, Machine}
    process::MarkovRewardProcess
    policy::MarkovPolicy
    io_config::IoConfig
    current_state::Union{Nothing, MarkovState}
end

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, player_configs::Dict)
    io_config = IoConfig(player_configs)
    state_to_value = read_values_file(io_config.values)
    return TemporalDifferencePlayer(TPolicy, team, state_to_value, Dict{UInt64, Float64}(), player_configs)
end
TemporalDifferencePlayer(team::Symbol, player_configs::Dict) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, player_configs)

TemporalDifferencePlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, player_configs)

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, player_configs)
    io_config = IoConfig(player_configs)
    machine = try_load_model_from_csv(io_config)
    machine_public = nothing

    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = TPolicy(machine)
    TemporalDifferencePlayer(Player(team), machine, machine_public, process, policy, io_config, nothing)
end

function Base.deepcopy(player::MutatedEmpathRobotPlayer)
    return MutatedEmpathRobotPlayer(deepcopy(player.player), player.machine, player.machine_public, deepcopy(player.mutation), player.io_config) 
end

function Base.deepcopy(player::TemporalDifferencePlayer)
    # Note, we deepcopy only the player data, while the RL data should persist in order to pass updates the state info properly
    return TemporalDifferencePlayer(
        deepcopy(player.player), 
        player.machine, 
        player.machine_public, 
        player.process, 
        player.policy, 
        player.io_config,
        deepcopy(player.current_state)
    )
end

#EmpathRobotPlayer(team::Symbol, player_configs::Dict) = EmpathRobotPlayer(team, IoConfig(player_configs).features)
#EmpathRobotPlayer(team::Symbol) = EmpathRobotPlayer(team, player_configs)
function EmpathRobotPlayer(team::Symbol, player_configs::Dict) 
    io_config = IoConfig(player_configs)
    EmpathRobotPlayer(
        Player(team), 
        try_load_model_from_csv(io_config),
        nothing
    )
end

function Base.deepcopy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(deepcopy(player.player), player.machine, player.machine_public)
end

inner_player(player::EmpathRobotPlayer)::Player = p -> p.player
inner_player(player::MutatedEmpathRobotPlayer)::Player = p -> p.player
inner_player(player::TemporalDifferencePlayer)::Player = p -> p.player
