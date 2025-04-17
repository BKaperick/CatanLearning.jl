module CatanLearning
using Logging
using Profile
using BenchmarkTools

import MLJModelInterface
const MMI = MLJModelInterface

import Catan
import Catan: Player, PlayerPublicView, PlayerType, RobotPlayer, DefaultRobotPlayer, Game, Board, 
get_player_config

include("structs.jl")
include("learning/naive_model.jl")
include("learning/production_model.jl")
include("players/structs.jl")
include("helpers.jl")
include("mutation_rule_library.jl")
include("evolution.jl")

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
function run(T::Type, configs::Dict, player_configs::Dict)
    #global_logger(logger)
    #global_logger()
    player_constructors = Dict([
        :Blue => (mutation) -> T(:Blue, player_configs), 
        :Green => (mutation) -> T(:Green, player_configs), 
        :Cyan => (mutation) -> T(:Cyan, player_configs), 
        :Yellow => (mutation) -> T(:Yellow, player_configs)
    ])
    run(player_constructors, configs)
end

function run_validation()
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
    master_state_to_value = read_values_file(player_configs["STATE_VALUES"])::Dict{UInt64, Float64}
    new_state_to_value = Dict{UInt64, Float64}()
    player_constructors = Dict([
        :Blue => (mutation) -> TemporalDifferencePlayer(
                                    MaxRewardPlusValueMarkovPolicy, 
                                    :Blue, 
                                    master_state_to_value, 
                                    new_state_to_value
                                    ), 
        :Green => (mutation) -> Catan.DefaultRobotPlayer(:Green), 
        :Cyan => (mutation) -> Catan.DefaultRobotPlayer(:Cyan), 
        :Yellow => (mutation) -> Catan.DefaultRobotPlayer(:Yellow)
    ])
    run(player_constructors)
end

function run_validation_ml()
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
    player_constructors = Dict([
        :Blue => (mutation) -> EmpathRobotPlayer(:Blue, player_configs), 
        :Green => (mutation) -> Catan.DefaultRobotPlayer(:Green, player_configs), 
        :Cyan => (mutation) -> Catan.DefaultRobotPlayer(:Cyan, player_configs), 
        :Yellow => (mutation) -> Catan.DefaultRobotPlayer(:Yellow, player_configs)
    ])
    run(player_constructors)
end

function run_explore()
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
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

function run(create_players::Function, configs)
    tourney = Tournament(configs, :Sequential)
    run_tournament(tourney, create_players, configs)
end
function run(player_schemas::Vector, configs)
    tourney = Tournament(configs, :Sequential)
    run_tournament(tourney, player_schemas, configs)
end

function run(player_constructors::Dict, configs)
    tourney = Tournament(configs, :Sequential)
    if any([typeof(c(Dict())) <: MutatedEmpathRobotPlayer for (t,c) in collect(player_constructors)])
        run_mutating_tournament(tourney, player_constructors, configs)
    end
end
end
