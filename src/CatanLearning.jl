module CatanLearning
using Logging
using Profile
using BenchmarkTools


# Suppress all normal logs
#logger = ConsoleLogger(stderr, Logging.Warn)
logger = ConsoleLogger(stderr, Logging.Info)
old = global_logger(logger)

#using Catan
import Catan
import Catan: Player, PlayerPublicView, PlayerType, RobotPlayer, DefaultRobotPlayer, Game, Board#, configs, player_configs, logger

println("DIR: $(@__DIR__)")
#global (configs, player_configs, logger) = Catan.reset_configs(joinpath(@__DIR__, "../Configuration.toml"))
#Catan.parse_configs("Configuration.toml")
#println("player configs in CL: $player_configs")

#include("../main.jl")
#include("../apis/player_api.jl")
include("constants.jl")
include("structs.jl")
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

Catan.configs["SAVE_GAME_TO_FILE"] = false


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

function run()
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
    run(EmpathRobotPlayer, configs, player_configs)
    #run(MutatedEmpathRobotPlayer, configs, player_configs)
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
    #global_logger(NullLogger())
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
    #global_logger(logger)
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
    else
        player_configs = configs["PlayerSettings"]
        create_players_pre = [eval(Meta.parse("() -> $(p[1])($(p[2]), player_configs)")) for p in player_constructors]
        create_players = () -> [p() for p in create_players_pre]
        run_tournament(tourney, create_players, configs)
    end
end

#run_benchmark() => run_benchmark(Catan.DefaultRobotPlayer)
function run_benchmark(player_type)
    global (configs, player_configs, logger) = Catan.parse_configs("Configuration.toml")
    global_logger(NullLogger())
    map_file = "./data/benchmark_map.csv"
    map = Catan.generate_random_map(map_file)
    team_to_player = Dict([
        :Blue => player_type(:Blue, player_configs), 
        :Green => player_type(:Green, player_configs), 
        :Cyan => player_type(:Cyan, player_configs), 
        :Yellow => player_type(:Yellow, player_configs)
    ])
    teams = collect(keys(team_to_player))
    players = collect(values(team_to_player))
     
    winners = init_winners(teams)
    #@profile do_tournament_one_game!(tourney, player_constructors); Profile.print(noisefloor = 2.0, combine = true)
    #@btime do_tournament_one_game!($tourney, $player_constructors)
    return @benchmark do_tournament_one_game!($winners, $players, $configs)
end

logger = ConsoleLogger(stderr, Logging.Warn)
old = global_logger(logger)
#=
if length(ARGS) > 0
    global_logger(NullLogger())
    global FEATURES_FILE = "features_$(ARGS[1]).csv"
    println("Setting global features file to $FEATURES_FILE")
    run(Catan.DefaultRobotPlayer)
else
    global FEATURES_FILE = "features_$(rand(1:100_000)).csv"
end
=#
end
