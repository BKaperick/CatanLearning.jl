module CatanLearning
using Logging
using Profile
using BenchmarkTools
#using Catan
import Catan
import Catan: Player, PlayerPublicView, PlayerType, RobotPlayer, DefaultRobotPlayer, Game, Board

#include("../main.jl")
#include("../apis/player_api.jl")
include("constants.jl")
include("structs.jl")
include("players/structs.jl")
include("helpers.jl")
include("mutation_rule_library.jl")
include("evolution.jl")
include("players/learning_player_base.jl")
include("players/mutated_robot_player.jl")
include("players/temporal_difference_player.jl")
include("io.jl")

include("tournaments.jl")



# Suppress all normal logs
logger = ConsoleLogger(stderr, Logging.Warn)
global_logger(logger)
SAVE_GAME_TO_FILE = false
#SAVEFILEIO = open(SAVEFILE, "a")


function run(T::MutatedEmpathRobotPlayer)
    player_constructors = Dict([
        :Blue => (mutation) -> T(:Blue, mutation), 
        :Green => (mutation) -> T(:Green, mutation), 
        :Cyan => (mutation) -> T(:Cyan, mutation), 
        :Yellow => (mutation) -> T(:Yellow, mutation)
    ])
    run(player_constructors)
end
function run(T::Type)
    player_constructors = Dict([
        :Blue => (mutation) -> T(:Blue), 
        :Green => (mutation) -> T(:Green), 
        :Cyan => (mutation) -> T(:Cyan), 
        :Yellow => (mutation) -> T(:Yellow)
    ])
    run(player_constructors)
end

function run()
    run(MutatedEmpathRobotPlayer)
end
function run_validation()
    master_state_to_value = read_values_file(IoConfig().values)::Dict{UInt64, Float64}
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

function run_explore()

    master_state_to_value = read_values_file(IoConfig().values)::Dict{UInt64, Float64}
    new_state_to_value = Dict{UInt64, Float64}()
    player_maker = team -> ((mutation) -> TemporalDifferencePlayer(
                                    MaxRewardPlusValueMarkovPolicy, 
                                    team, 
                                    master_state_to_value, 
                                    new_state_to_value
                                   )
                           )
    player_constructors = Dict([
                                :Blue => player_maker(:Blue), 
                                :Green => player_maker(:Green), 
                                :Cyan => player_maker(:Cyan), 
                                :Yellow => player_maker(:Yellow)
    ])
    run(player_constructors)
end

function run(player_constructors::Dict)
    # Number of games to play per map
    # Number of maps to generate
    # Number of epochs (1 epoch is M*N games) to run
    #tourney = Tournament(2, 2, 2, :Sequential)
    tourney = Tournament(100, 1000, 1, :Sequential)
    #tourney = Tournament(20,8,20, :FiftyPercentWinnerStays)
    #tourney = Tournament(5,4,10, :SixtyPercentWinnerStays)
    if any([typeof(c(Dict())) <: MutatedEmpathRobotPlayer for (t,c) in collect(player_constructors)])
        run_mutating_tournament(tourney, player_constructors)
    else
        #run_tournament(tourney, player_constructors)
        #@profile run_tournament(tourney, player_constructors); Profile.print(noisefloor = 2.0, combine = true)
        #@btime run_tournament($tourney, $player_constructors)
        println(@benchmark run_tournament($tourney, $player_constructors))
    end
end

#run_benchmark() => run_benchmark(Catan.DefaultRobotPlayer)
function run_benchmark(player_type)
    map_file = "./data/benchmark_map.csv"
    map = Catan.generate_random_map(map_file)
    team_to_player = Dict([
        :Blue => player_type(:Blue), 
        :Green => player_type(:Green), 
        :Cyan => player_type(:Cyan), 
        :Yellow => player_type(:Yellow)
    ])
    teams = collect(keys(team_to_player))
    players = collect(values(team_to_player))
     
    winners = init_winners(teams)
    do_tournament_one_game!(winners, players, map_file)
    #@benchmark do_tournament_one_game!($winners, $players, $map_file)
end

end
