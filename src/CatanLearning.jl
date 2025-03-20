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


function run()
    team_to_mutation = Dict([
        :Blue => Dict(),
        :Green => Dict(),
        :Cyan => Dict(),
        :Yellow => Dict()
    ])

    # Number of games to play per map
    # Number of maps to generate
    # Number of epochs (1 epoch is M*N games) to run
    tourney = Tournament(2,2,2, :Sequential)
    #tourney = Tournament(20,8,20, :FiftyPercentWinnerStays)
    #tourney = Tournament(5,4,10, :SixtyPercentWinnerStays)

    run_tournament(typeof(MutatedEmpathRobotPlayer), tourney, team_to_mutation)
end

function run_tournament(T, tourney, team_to_mutation)
    winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in collect(keys(team_to_mutation))])
    winners[nothing] = 0
    map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
    for k=1:tourney.epochs
        for (w,v) in winners
            winners[w] = 0
        end
        for (player,mt) in team_to_mutation
            println("$(player): $(print_mutation(mt))")
        end
        for j=1:tourney.maps_per_epoch
            map = Catan.generate_random_map(map_file)
            for i=1:tourney.games_per_map
                game = Game(generate_players(team_to_mutation))
                println("starting game $(game.unique_id)")
                _,winner = Catan.run(game, map_file)

                w = winner
                if winner != nothing
                    w = winner.player.team
                end
                winners[w] += 1
                if winner != nothing
                    println("Game $i: $(winner.player.team)")
                end
            end
        end
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
end
