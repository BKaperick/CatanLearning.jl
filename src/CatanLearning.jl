module CatanLearning
using Logging
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



# Suppress all normal logs
logger = ConsoleLogger(stderr, Logging.Warn)
global_logger(logger)
SAVE_GAME_TO_FILE = false
#SAVEFILEIO = open(SAVEFILE, "a")


function run(T::Type)
    player_constructors = Dict([
        :Blue => (mutation) -> T(:Blue, mutation), 
        :Green => (mutation) -> T(:Green, mutation), 
        :Cyan => (mutation) -> T(:Cyan, mutation), 
        :Yellow => (mutation) -> T(:Yellow, mutation)
    ])
    run(player_constructors)
end

function run()
    run(MutatedEmpathRobotPlayer)
end

function run(player_constructors::Dict)
    # Number of games to play per map
    # Number of maps to generate
    # Number of epochs (1 epoch is M*N games) to run
    tourney = Tournament(2,2,2, :Sequential)
    #tourney = Tournament(20,8,20, :FiftyPercentWinnerStays)
    #tourney = Tournament(5,4,10, :SixtyPercentWinnerStays)
    if any([typeof(c(Dict())) <: MutatedEmpathRobotPlayer for (t,c) in collect(player_constructors)])
        run_mutating_tournament(tourney, player_constructors)
    else
        run_tournament(tourney, player_constructors)
    end
end

function do_tournament_one_epoch(tourney, teams, map_file, player_constructors)
    do_tournament_one_epoch(tourney, teams, map_file, player_constructors, Dict([(t,Dict()) for t in teams]A))
end
function do_tournament_one_epoch(tourney, teams, map_file, player_constructors, team_to_mutation)
    winners = init_winners(teams)
    for j=1:tourney.maps_per_epoch
        map = Catan.generate_random_map(map_file)
        for i=1:tourney.games_per_map
            game = Game([player_constructors[t](team_to_mutation[t]) for t in teams])
            do_tournament_one_game(tourney, i, j, winners, game, map_file)
        end
    end
end

function do_tournament_one_game(tourney, i, j, winners, game, map_file)
    println("starting game $(game.unique_id)")
    _,winner = Catan.run(game, map_file)

    w = winner
    if winner != nothing
        w = winner.player.team
    end
    winners[w] += 1

    if winner != nothing
        println("Game $((j - 1)*tourney.games_per_map + i) / $(tourney.maps_per_epoch * tourney.games_per_map): $(winner.player.team)")
        println("winner: $(winner.player.team)")
    end
    return winner
end

function init_winners(teams)
    winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
    winners[nothing] = 0
    return winners
end

function run_mutating_tournament(tourney, player_constructors)
    teams = collect(keys(player_constructors))
    team_to_mutation = Dict([(t, Dict()) for t in teams])
    map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        do_tournament_one_epoch(tourney, teams, map_file, player_constructors, team_to_mutation)
        
        # Don't assign new mutations on the last one so we can see the results
        if k < tourney.epochs
            ordered_winners = order_winners(winners)
            apply_mutation_rule![tourney.mutation_rule](team_to_mutation, ordered_winners)
        end
    end
    println(winners)

    for (player,mt) in team_to_mutation
        println("$(player): $(print_mutation(mt))")
    end
end

function run_tournament(tourney, player_constructors)
    teams = collect(keys(player_constructors))
    map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        do_tournament_one_epoch(tourney, teams, map_file, player_constructors)
    end
    println(winners)
end

end
