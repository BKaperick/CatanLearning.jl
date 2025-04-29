function do_tournament_one_epoch(tourney, teams, configs, player_constructors::Dict)
    do_tournament_one_epoch(tourney, teams, configs, player_constructors, Dict([(t,Dict()) for t in teams]))
end
function do_tournament_one_epoch(tourney, teams, configs, player_constructors::Dict, team_to_mutation::Dict)
    create_players = () -> [player_constructors[t](team_to_mutation[t]) for t in teams]
end
function do_tournament_one_epoch(tourney, teams, configs, players_schema::Vector)
    players = Catan.get_known_players()
    println(players)

    f = () -> [players[p[1]](Symbol(p[2]), configs) for p in players_schema]
    do_tournament_one_epoch(tourney, teams, configs, f)
end

function do_tournament_one_epoch(tourney, teams, configs, create_players::Function)
    winners = init_winners(teams)
    for j=1:tourney.maps_per_epoch
        map = Catan.generate_random_map(configs["MAP_FILE"])
        for i=1:tourney.games_per_map
            players = create_players()
            winner = do_tournament_one_game!(winners, players, configs)
            g_num = (j - 1)*tourney.games_per_map + i
            if winner !== nothing
                #println("winner: $(winner.player.team)")
            else
                #println("winner: noone")
            end
            if g_num % 100 == 0
                println("Game $(g_num) / $(tourney.maps_per_epoch * tourney.games_per_map)")
            end
        end
    end
    order_winners(winners)
end

function do_tournament_one_game!(winners, players, configs)
    game = Game(players, configs)
    #println("starting game $(game.unique_id)")
    _,winner = Catan.run(game)

    w = winner
    if winner !== nothing
        w = winner.player.team
        #@warn "$w won"
    end
    winners[w] += 1

    return winner
end

function init_winners(teams)
    winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
    winners[nothing] = 0
    return winners
end

function run_mutating_tournament(tourney, player_constructors, configs)
    teams = configs["TEAMS"]
    team_to_mutation = Dict([(t, Dict()) for t in teams])
    configs["MAP_FILE"] = "./data/_temp_map_file.csv"
    for k=1:tourney.epochs
        ordered_winners = do_tournament_one_epoch(tourney, teams, configs, player_constructors, team_to_mutation)
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

function run_tournament(tourney, player_schemas::Vector, configs)
    println(player_schemas[1])
    teams = [Symbol(t) for t in configs["TEAMS"]]
    configs["MAP_FILE"] = "./data/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        println("epoch $k / $(tourney.epochs)")
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs, player_schemas)
        #println(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
    println(winners)
end
function run_tournament(tourney, create_players::Function, configs)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    configs["MAP_FILE"] = "./data/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs, create_players)
        #println(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
end
