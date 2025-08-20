module CatanLearning
using Logging
using Profile
using BenchmarkTools
using DataStructures

import MLJModelInterface
const MMI = MLJModelInterface

import Catan
import Catan: Player, PlayerPublicView, PlayerType, RobotPlayer, DefaultRobotPlayer, Game, Board, Map,
get_player_config

export LearningPlayer, EmpathRobotPlayer, HybridPlayer, compute_features, get_state_score

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
    Catan.add_player_to_register("TemporalDifferencePlayer", (t,c) -> TemporalDifferencePlayer(t,c))
    Catan.add_player_to_register("HybridPlayer", (t,c) -> HybridPlayer(t,c))

    # Upsert the configs from this package
    default_config_path = joinpath(@__DIR__, "..", "DefaultConfiguration.toml")
    Catan.update_default_configs(default_config_path)
end

function run(configs)
    tourney = Tournament(configs, :Sequential)
    run_tournament(tourney, configs)
end

function descend_logger(configs, logger_prefix)
    main_logger = global_logger()
    descended_logger = create_descended_logger(configs, logger_prefix)
    global_logger(descended_logger)
    return main_logger
end

function create_descended_logger(configs, logger_prefix)
    logging_config = configs["LogSettings"]
    level = get(logging_config, "$(logger_prefix)_LOG_LEVEL", "Warn")
    out = get(logging_config, "$(logger_prefix)_LOG_OUTPUT", logging_config["LOG_OUTPUT"])
    descended_logger,_,__ = Catan.make_logger(level, out)
    return descended_logger
end
end