function do_tournament_one_epoch(tourney, teams, map_file, player_constructors::Dict, T::Type)
    do_tournament_one_epoch(tourney, teams, map_file, player_constructors, Dict([(t,Dict()) for t in teams]), T)
end
function do_tournament_one_epoch(tourney, teams, map_file, player_constructors::Dict, team_to_mutation::Dict, T::Type)
    winners = init_winners(teams)
    for j=1:tourney.maps_per_epoch
        map = Catan.generate_random_map(map_file)
        for i=1:tourney.games_per_map
            players = [player_constructors[t](team_to_mutation[t]) for t in teams]
            game = Game(players::Vector{T})
            game = Game(players)
            do_tournament_one_game(tourney, i, j, winners, game, map_file)
        end
    end
    print(winners)
    order_winners(winners)
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

function run_mutating_tournament(tourney, player_constructors, T=Vector{PlayerType})
    teams = collect(keys(player_constructors))
    team_to_mutation = Dict([(t, Dict()) for t in teams])
    map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
    for k=1:tourney.epochs
        ordered_winners = do_tournament_one_epoch(tourney, teams, map_file, player_constructors, team_to_mutation)
        println(ordered_winners)
        
        # Don't assign new mutations on the last one so we can see the results
        if k < tourney.epochs
            apply_mutation_rule![tourney.mutation_rule](team_to_mutation, ordered_winners)
        end
    end

    for (player,mt) in team_to_mutation
        println("$(player): $(print_mutation(mt))")
    end
end

function run_tournament(tourney, player_constructors, T=Vector{PlayerType})
    teams = collect(keys(player_constructors))
    map_file = "$(Catan.DATA_DIR)/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        epoch_winners = do_tournament_one_epoch(tourney, teams, map_file, player_constructors, T)
        println(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
end
