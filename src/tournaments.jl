function do_tournament_one_epoch(tourney, teams, configs, player_constructors::Dict)
    do_tournament_one_epoch(tourney, teams, configs, player_constructors, Dict([(t,Dict()) for t in teams]))
end
function do_tournament_one_epoch(tourney, teams, configs, player_constructors::Dict, team_to_mutation::Dict)
    create_players = () -> [player_constructors[t](team_to_mutation[t]) for t in teams]
end
function do_tournament_one_epoch(tourney, teams, configs, players_schema::Vector)
    players = Catan.get_known_players()
    @debug players

    f = () -> [players[p[1]](Symbol(p[2]), configs) for p in players_schema]
    do_tournament_one_epoch(tourney, teams, configs, f)
end

function do_tournament_one_epoch(tourney, teams, configs, create_players::Function)
    winners = init_winners(teams)
    for j=1:tourney.maps_per_epoch
        @info "map $j / $(tourney.maps_per_epoch)"
        do_tournament_one_map!(winners, tourney, configs, create_players)
    end
    order_winners(winners)
end

function do_tournament_one_epoch_async(channels, tourney, teams, configs, players_schema::Vector)
    players = Catan.get_known_players()

    create_players = () -> [players[p[1]](Symbol(p[2]), configs) for p in players_schema]
    @debug "players loaded"
    for j=1:tourney.maps_per_epoch
        @info "game $j / $(tourney.maps_per_epoch)"
        do_tournament_one_map_async!(channels, tourney, configs, create_players)
    end
end

function do_tournament_one_map!(winners, tourney, configs, create_players)
    map = Catan.generate_random_map(configs["MAP_FILE"])
    for i=1:tourney.games_per_map
        players = create_players()
        do_tournament_one_game!(winners, map, players, configs)

        g_num = (j - 1)*tourney.games_per_map + i
        if g_num % 100 == 0
            println("Game $(g_num) / $(tourney.maps_per_epoch * tourney.games_per_map)")
        end
    end
end

function do_tournament_one_map_async!(channels, tourney, configs, create_players)
    @debug "running do_tournament_one_map_async!"
    map = Catan.generate_random_map()
    for i=1:tourney.games_per_map
        @info "game $i / $(tourney.games_per_map)"
        players = create_players()
        do_tournament_one_game_async!(channels, map, players, configs)
    end
end

function do_tournament_one_game!(winners, map, players, configs)
    game = Game(players, configs)
    board = Catan.read_map(map)
    _,winner = Catan.run(game)

    w = winner
    if winner !== nothing
        w = winner.player.team
        #@warn "$w won"
    end
    winners[w] += 1

    return winner
end

function do_tournament_one_game_async!(channels, map, players, configs)
    println("running one game")
    game = Game(players, configs)
    board = Catan.read_map(configs, map)
    _,winner = Catan.run_async(channels, game, board)
    #=
    #channels[:winners]
    w = winner
    if winner !== nothing
        w = winner.player.team
        #@warn "$w won"
    end
    winners[w] += 1

    return winner
    =#
    return
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
        @info ordered_winners
        
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
    teams = [Symbol(t) for t in configs["TEAMS"]]
    configs["MAP_FILE"] = "./data/_temp_map_file.csv"
    winners = init_winners(teams)
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs, player_schemas)
        #println(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
    @debug winners
end

function run_tournament_async(configs)
    tourney = Tournament(configs, :Sequential)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    player_schemas = Catan.read_player_constructors_from_config(configs["PlayerSettings"])

    channels = Catan.read_channels_from_config(configs)
    
    println("Runnin dis tourney")
    #_run_tournament_async(channels, tourney, player_schemas, configs, teams)
    #do_post_game_consume!(channels, configs)
    
    #Threads.@spawn _run_tournament_async(channels, tourney, player_schemas, configs, teams))
    #errormonitor(Threads.@spawn CatanLearning.do_post_game_consume!(channels, game, board, players, player, winner))
    #t1 = Threads.@spawn _run_tournament_async(channels, tourney, player_schemas, configs, teams)
    #fetch(t1)

    #t2 = Threads.@spawn CatanLearning.do_post_game_consume!(channels, configs)
    #fetch(t2)

    #=
    @sync while ~isempty(channels[:main])
        @async consume_channel!(channels[:main], configs["PlayerSettings"]["FEATURES"])
    end
    close(channels[:main])
    =#

    game_tasks = _run_tournament_async(channels, tourney, player_schemas, configs, teams)
    
    for t in game_tasks
        fetch(t)
        consume_remaining_features!(channels, configs)
    end
    
    #=
    while ~isempty(channels[:public]) || ~isempty(channels[:main])
        t1 = Threads.@spawn consume_channel!(channels[:main], configs["PlayerSettings"]["FEATURES"])
        consume_channel!(channels[:public], configs["PlayerSettings"]["PUBLIC_FEATURES"])
        fetch(t1)
    end
    consume_remaining_features!(channels, configs)
    =#

    close(channels[:main])
    close(channels[:public])
end

function consume_remaining_features!(channels, configs)
    while ~isempty(channels[:public]) || ~isempty(channels[:main])
        t1 = Threads.@spawn consume_channel!(channels[:main], configs["PlayerSettings"]["FEATURES"])
        consume_channel!(channels[:public], configs["PlayerSettings"]["PUBLIC_FEATURES"])
        fetch(t1)
    end
end

#=
function consume_feature_tasks(channels::Dict{Symbol, Channel}, game::Game, board::Board, players::Vector{PlayerType}, player::Catan.DefaultRobotPlayer, winner::Union{PlayerType, Nothing}))
    

    # Put game data on channel
    @async do_post_game_produce!(channels, game, board, game.players, winner)
end
=#

function _run_tournament_async(channels, tourney, player_schemas::Vector, configs, teams)
    tasks = []
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        push!(tasks, Threads.@spawn do_tournament_one_epoch_async(channels, tourney, teams, configs, player_schemas))
    end
    #=
    for (n,c) in channels
        close(c)
    end
    =#
    return tasks
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
