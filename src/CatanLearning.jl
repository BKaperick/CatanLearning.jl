module CatanLearning
using Logging
using Profile
using BenchmarkTools
using DataStructures

import MLJModelInterface
const MMI = MLJModelInterface

import Catan
import Catan: Player, PlayerPublicView, PlayerType, RobotPlayer, DefaultRobotPlayer, Game, Board, 
get_player_config

function toggleprint(str)
    #println(str)
end

include("structs.jl")
include("learning/naive_model.jl")
include("learning/production_model.jl")
include("players/structs.jl")
include("helpers.jl")
include("mutation_rule_library.jl")

include("learning/feature_computation.jl")

include("players/learning_player_base.jl")
include("players/mutated_robot_player.jl")
include("players/temporal_difference_player.jl")
include("io.jl")

include("tournaments.jl")

function __init__()
    Catan.add_player_to_register("EmpathRobotPlayer", (t,c) -> EmpathRobotPlayer(t,c))
    Catan.add_player_to_register("MutatedEmpathRobotPlayer", (t,c) -> MutatedEmpathRobotPlayer(t,c))
    Catan.add_player_to_register("TemporalDifferencePlayer", (t,c) -> TemporalDifferencePlayer(t,c))
    Catan.add_player_to_register("HybridPlayer", (t,c) -> HybridPlayer(t,c))

    # Upsert the configs from this package
    default_config_path = joinpath(@__DIR__, "..", "DefaultConfiguration.toml")
    Catan.update_default_configs(default_config_path)
end

function run(T::MutatedEmpathRobotPlayer, configs::Dict, player_configs::Dict)
    player_constructors = Dict([
        :Blue => (mutation) -> T(:Blue, mutation, player_configs), 
        :Green => (mutation) -> T(:Green, mutation, player_configs), 
        :Cyan => (mutation) -> T(:Cyan, mutation, player_configs), 
        :Yellow => (mutation) -> T(:Yellow, mutation, player_configs)
    ])
    run(player_constructors, configs)
end

function run_explore()
    configs = Catan.parse_configs("Configuration.toml")
    player_configs = configs["PlayerSettings"]
    master_state_to_value = read_values_file(player_configs["STATE_VALUES"])::Dict{UInt64, Float64}
    new_state_to_value = Dict{UInt64, Float64}()
    player_maker = team -> ((mutation) -> TemporalDifferencePlayer(
                                    MaxRewardPlusValueMarkovPolicy, 
                                    team, 
                                    master_state_to_value, 
                                    new_state_to_value,
                                    player_configs
                                   )
                           )
    player_constructors = Dict([
                                :Blue => player_maker(:Blue), 
                                :Green => player_maker(:Green), 
                                :Cyan => player_maker(:Cyan), 
                                :Yellow => player_maker(:Yellow)
    ])
    run(player_constructors, configs)
end

function run(configs)
    tourney = Tournament(configs, :Sequential)
    run_tournament(tourney, configs)
end

function run(player_constructors::Dict, configs)
    tourney = Tournament(configs, :Sequential)
    if any([typeof(c(Dict())) <: MutatedEmpathRobotPlayer for (t,c) in collect(player_constructors)])
        run_mutating_tournament(tourney, player_constructors, configs)
    end
end

function descend_logger(configs, logger_prefix)
    logging_config = configs["LogSettings"]
    level = get(logging_config, "$(logger_prefix)_LOG_LEVEL", "Warn")
    out = get(logging_config, "$(logger_prefix)_LOG_OUTPUT", logging_config["LOG_OUTPUT"])
    descended_logger,_,__ = Catan.make_logger(level, out)
    @debug "setting logger to $level as we enter $logger_prefix environment"
    main_logger = global_logger()
    global_logger(descended_logger)
    return main_logger
end
end
