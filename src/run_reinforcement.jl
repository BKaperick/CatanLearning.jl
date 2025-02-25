using Logging

using Catan
import Catan

include("structs.jl")
include("constants.jl")
include("helpers.jl")
include("players/structs.jl")
include("players/temporal_difference_player.jl")


teams = [
   :Blue,
   :Green,
   :Cyan,
   :Yellow
]

# Suppress all normal logs
logger = ConsoleLogger(stderr, Logging.Warn)
global_logger(logger)
SAVE_GAME_TO_FILE = false
#SAVEFILEIO = open(SAVEFILE, "a")

map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
winners[nothing] = 0

# Number of games to play per map
# Number of maps to generate
# Number of epochs (1 epoch is M*N games) to run
tourney = Tournament(2,2,2, :Sequential)
#tourney = Tournament(20,8,20, :FiftyPercentWinnerStays)
#tourney = Tournament(5,4,10, :SixtyPercentWinnerStays)

master_state_to_value = read_values_file(IoConfig().values)::Dict{UInt64, Float64}
#master_state_to_value = Dict{UInt64, Float64}()

for k=1:tourney.epochs
    for (w,v) in winners
        winners[w] = 0
    end
    for j=1:tourney.maps_per_epoch
        map = Catan.generate_random_map(map_file)
        for i=1:tourney.games_per_map
            game = Game([TemporalDifferencePlayer(t, master_state_to_value) for t in teams])
            println("starting game $(game.unique_id)")
            _,winner = initialize_and_do_game!(game, map_file)

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
    end
end
println(winners)
println(length(master_state_to_value))
println(collect(master_state_to_value)[1:10])
