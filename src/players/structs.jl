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
    configs::Dict
end

MutatedEmpathRobotPlayer(team::Symbol, player_configs::Dict) = MutatedEmpathRobotPlayer(
    team, Dict{Symbol, AbstractFloat}(), player_configs)

function MutatedEmpathRobotPlayer(team::Symbol, mutation::Dict, player_configs::Dict) 
    MutatedEmpathRobotPlayer(
    Player(team), 
    try_load_model_from_csv(player_configs),
    nothing,
    mutation,
    player_configs)
end

mutable struct TemporalDifferencePlayer <: LearningPlayer
    player::Player
    machine::Machine
    machine_public::Union{Nothing, Machine}
    process::MarkovRewardProcess
    policy::MarkovPolicy
    configs::Dict
    current_state::Union{Nothing, MarkovState}
end

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, player_configs::Dict)
    state_to_value = read_values_file(player_configs["STATE_VALUES"])
    return TemporalDifferencePlayer(TPolicy, team, state_to_value, Dict{UInt64, Float64}(), player_configs)
end
TemporalDifferencePlayer(team::Symbol, player_configs::Dict) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, player_configs)

TemporalDifferencePlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}) = TemporalDifferencePlayer(MaxRewardMarkovPolicy, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, player_configs)

function TemporalDifferencePlayer(TPolicy::Type, team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64}, player_configs)
    machine = try_load_model_from_csv(player_configs)
    machine_public = nothing

    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = TPolicy(machine)
    TemporalDifferencePlayer(Player(team), machine, machine_public, process, policy, player_configs, nothing)
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

function EmpathRobotPlayer(team::Symbol, player_configs::Dict) 
    EmpathRobotPlayer(
        Player(team), 
        try_load_model_from_csv(player_configs),
        nothing
    )
end

function Base.deepcopy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(deepcopy(player.player), player.machine, player.machine_public)
end

inner_player(player::EmpathRobotPlayer)::Player = p -> p.player
inner_player(player::MutatedEmpathRobotPlayer)::Player = p -> p.player
inner_player(player::TemporalDifferencePlayer)::Player = p -> p.player

Catan.add_player_to_register("EmpathRobotPlayer", (t,c) -> EmpathRobotPlayer(t,c))
