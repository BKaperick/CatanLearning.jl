using Logging

using Catan
import Catan

include("structs.jl")
include("constants.jl")
#include("helpers.jl")
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

#map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
winners[nothing] = 0

# Number of games to play per map
# Number of maps to generate
# Number of epochs (1 epoch is M*N games) to run
tourney = Tournament(10,2000,1, :Sequential)
#tourney = Tournament(20,8,20, :FiftyPercentWinnerStays)
#tourney = Tournament(5,4,10, :SixtyPercentWinnerStays)

master_state_to_value = read_values_file(player_configs["STATE_VALUES"])::Dict{UInt64, Float64}
new_state_to_value = Dict{UInt64, Float64}()
start_length = length(master_state_to_value)
println("starting states known: $(start_length)")

for k=1:tourney.epochs
    for (w,v) in winners
        winners[w] = 0
    end
    for j=1:tourney.maps_per_epoch
        map = Catan.generate_random_map(map_file)
        for i=1:tourney.games_per_map
            @assert length(new_state_to_value) == 0
            @assert isempty(intersect(keys(new_state_to_value), keys(master_state_to_value)))
            game = Game(Vector{TemporalDifferencePlayer}([TemporalDifferencePlayer(t, master_state_to_value, new_state_to_value) for t in teams]))
            #println("starting game $(game.unique_id)")
            @assert length(new_state_to_value) == 0
            @assert isempty(intersect(keys(new_state_to_value), keys(master_state_to_value)))
            _,winner = Catan.run(game, map_file)
            @assert length(new_state_to_value) == 0
            @assert isempty(intersect(keys(new_state_to_value), keys(master_state_to_value)))

            w = winner
            if winner != nothing
                w = winner.player.team
            end
            winners[w] += 1
            if winner != nothing
                println("Game $((j - 1)*tourney.games_per_map + i) / $(tourney.maps_per_epoch * tourney.games_per_map): $(winner.player.team)")
            end

            end_length = length(master_state_to_value)
            println("ending states known: $(end_length) ($(end_length - start_length) new states visited, so now ~$(100*end_length / (32^5))% of total state space has been explored)")
        end
    end
    # Don't assign new mutations on the last one so we can see the results
    if k < tourney.epochs
        ordered_winners = order_winners(winners)
    end



end

return write_values_file(player_configs["STATE_VALUES"], master_state_to_value)
println(winners)
