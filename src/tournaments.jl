
function initialize_tournament(configs::Dict)
    if configs["WRITE_FEATURES"] == true
        @info "Intializing player feature files"
        f = get_features()
        pf = get_public_features()
        for team in configs["TEAMS"]
            f_file_name = get_player_config(configs, "FEATURES", team)
            pf_file_name = get_player_config(configs, "PUBLIC_FEATURES", team)
            println("checking in $f_file_name")
            write_features_header_if_needed(f_file_name, f, configs)
            println("checking in $pf_file_name")
            write_features_header_if_needed(pf_file_name, pf, configs)
        end
    end
end

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
        do_tournament_one_map!(winners, tourney, configs, create_players, j)
    end
    order_winners(winners)
end

function do_tournament_one_epoch_async(channels, tourney, teams, configs, players_schema::Vector)
    players = Catan.get_known_players()

    create_players = () -> [players[p[1]](Symbol(p[2]), configs) for p in players_schema]
    for j=1:tourney.maps_per_epoch
        @info "map $j / $(tourney.maps_per_epoch)"
        do_tournament_one_map_async!(channels, tourney, configs, create_players)

        #TODO better to control this with yield here or just implicitly with the Channel buffer size?
        #yield()
    end
end

function do_tournament_one_map!(winners, tourney, configs, create_players, map_num)
    map = Catan.generate_random_map()
    for i=1:tourney.games_per_map
        players = create_players()
        do_tournament_one_game!(winners, map, players, configs)

        g_num = (map_num - 1)*tourney.games_per_map + i
        if g_num % 100 == 0
            toggleprint("Game $(g_num) / $(tourney.maps_per_epoch * tourney.games_per_map)")
        end
    end
end

function do_tournament_one_map_async!(channels, tourney, configs, create_players)
    @debug "running do_tournament_one_map_async!"
    map = Catan.generate_random_map()
    for i=1:tourney.games_per_map
        @debug "game $i / $(tourney.games_per_map)"
        players = create_players()
        do_tournament_one_game_async!(channels, map, players, configs)
        yield()
    end
end

function do_tournament_one_game!(winners, map, players, configs)
    game = Game(players, configs)
    board = Catan.read_map(configs, map)
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
    game = Game(players, configs)
    board = Catan.read_map(configs, map)
    @debug "starting game $(game.unique_id)"
    main_logger = global_logger()
    global_logger(NullLogger())
    _,winner = Catan.run_async(channels, game, board)
    global_logger(main_logger)
    @debug "finished game $(game.unique_id)"
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
    configs["SAVE_MAP"] = "./data/_temp_map_file.csv"
    for k=1:tourney.epochs
        ordered_winners = do_tournament_one_epoch(tourney, teams, configs, player_constructors, team_to_mutation)
        @info ordered_winners
        
        # Don't assign new mutations on the last one so we can see the results
        if k < tourney.epochs
            apply_mutation_rule![tourney.mutation_rule](team_to_mutation, ordered_winners)
        end
    end

    for (player,mt) in team_to_mutation
        toggleprint("$(player): $(print_mutation(mt))")
    end
end

function run_tournament(configs::Dict)
    initialize_tournament(configs)
    tourney = Tournament(configs, :Sequential)
    player_schemas = Catan.read_player_constructors_from_config(configs["PlayerSettings"])
    teams = [Symbol(t) for t in configs["TEAMS"]]
    winners = init_winners(teams)
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs, player_schemas)
        #toggleprint(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
    @debug winners
end

function run_tournament_async(configs)
    initialize_tournament(configs)
    tourney = Tournament(configs, :Sequential)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    player_schemas = Catan.read_player_constructors_from_config(configs["PlayerSettings"])

    channels = Catan.read_channels_from_config(configs)
    
    toggleprint("Runnin dis tourney")

    data_points = 4*(tourney.games_per_map * tourney.maps_per_epoch * tourney.epochs)
    @info "Running tournament of $(data_points/4) games in total"
    
    @sync begin
        @async _run_tournament_async(channels, tourney, player_schemas, configs, teams)
        @async consume_feature_channel!(channels[:main], data_points, configs["PlayerSettings"]["FEATURES"])
        @async consume_feature_channel!(channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"])
    end
    #=
    _run_tournament_async(channels, tourney, player_schemas, configs, teams)
    consume_feature_channel!(channels[:main], data_points, configs["PlayerSettings"]["FEATURES"])
    consume_feature_channel!(channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"])
    =#
end

function consume_feature_channel!(channel, count, key)
    for i=1:count
        consume_channel!(channel, key)
    end
    close(channel)
end

function _run_tournament_async(channels, tourney, player_schemas::Vector, configs, teams)
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        do_tournament_one_epoch_async(channels, tourney, teams, configs, player_schemas)
    end
end

function run_tournament(tourney, create_players::Function, configs)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    winners = init_winners(teams)
    for k=1:tourney.epochs
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs, create_players)
        #toggleprint(epoch_winners)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
end
