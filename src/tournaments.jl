
function initialize_tournament(configs::Dict)
    if configs["WRITE_FEATURES"] == true
        @info "Intializing player feature files"
        f = get_features()
        pf = get_public_features()
        for team in configs["TEAMS"]
            f_file_name = get_player_config(configs, "FEATURES", team)
            pf_file_name = get_player_config(configs, "PUBLIC_FEATURES", team)
            @debug "checking in $f_file_name"
            write_features_header_if_needed(f_file_name, f, configs)
            @debug "checking in $pf_file_name"
            write_features_header_if_needed(pf_file_name, pf, configs)
        end
    end
end

function do_tournament_one_epoch(tourney, teams, configs; create_players = Catan.create_players)::Vector{Tuple{Union{Nothing,Symbol}, Int}}
    winners = init_winners(teams)
    map_str = ""
    if !configs["Tournament"]["GENERATE_RANDOM_MAPS"]
        map_str = read(configs["LOAD_MAP"], String)
    end
    for j=1:tourney.maps_per_epoch
        @info "map $j / $(tourney.maps_per_epoch)"
        if configs["Tournament"]["GENERATE_RANDOM_MAPS"]
            do_tournament_one_map!(winners, tourney, configs, j; create_players = create_players)
        else
            do_tournament_one_map!(winners, tourney, configs, j, map_str; create_players = create_players)
        end
        @info winners
    end
    order_winners(winners)
end

function do_tournament_one_epoch_async(channels, tourney, teams, configs)
    for j=1:tourney.maps_per_epoch
        @info "map $j / $(tourney.maps_per_epoch)"
        do_tournament_one_map_async!(channels, tourney, configs)

        #TODO better to control this with yield here or just implicitly with the Channel buffer size?
        #yield()
    end
end


function do_tournament_one_map!(winners, tourney, configs, map_num::Integer; create_players = Catan.create_players)
    map = Catan.generate_random_map()
    do_tournament_one_map!(winners, tourney, configs, map_num, map; create_players)
end
function do_tournament_one_map!(winners, tourney, configs, map_num::Integer, map_str::AbstractString; create_players = Catan.create_players)
    
    function log_games_per_map(map_num, tourney, i)
        g_num = (map_num - 1)*tourney.games_per_map + i
        if g_num % 100 == 0
            toggleprint("Game $(g_num) / $(tourney.maps_per_epoch * tourney.games_per_map)")
        end
    end

    iter_logger = (tourney, i) -> log_games_per_map(map_num, tourney, i)
    do_tournament_one_map!(winners, tourney, configs, map_str, iter_logger; create_players)
end

function do_tournament_one_map!(winners, tourney, configs, map::AbstractString, iter_logger; create_players = Catan.create_players)
    for i=1:tourney.games_per_map
        players = create_players(configs)
        do_tournament_one_game!(winners, map, players, configs)
        iter_logger(tourney, i)
    end
end

function do_tournament_one_map_async!(channels, tourney, configs)
    @debug "running do_tournament_one_map_async!"
    map = Catan.generate_random_map()
    for i=1:tourney.games_per_map
        @debug "game $i / $(tourney.games_per_map)"
        players = Catan.create_players(configs)
        do_tournament_one_game_async!(channels, map, players, configs)
        yield()
    end
end

function do_tournament_one_game!(winners, map, players, configs)
    game = Game(players, configs)
    board = Catan.read_map(configs, map)
    main_logger = descend_logger(configs, "GAME")
    _,winner = Catan.run(game, board)
    global_logger(main_logger)
    @debug "finished game $(game.unique_id)"

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

function init_winners(teams)::Dict{Union{Symbol, Nothing}, Int}
    winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
    winners[nothing] = 0
    return winners
end

function run_tournament(configs::Dict)
    initialize_tournament(configs)
    tourney = Tournament(configs, :Sequential)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    winners = init_winners(teams)
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
    @info winners
end

function run_tournament_async(configs)
    initialize_tournament(configs)
    tourney = Tournament(configs, :Sequential)
    teams = [Symbol(t) for t in configs["TEAMS"]]

    channels = Catan.read_channels_from_config(configs)
    
    toggleprint("Runnin dis tourney")

    data_points = 4*(tourney.games_per_map * tourney.maps_per_epoch * tourney.epochs)
    @info "Running tournament of $(data_points/4) games in total"
    
    @sync begin
        @async _run_tournament_async(channels, tourney, configs, teams)
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

function _run_tournament_async(channels, tourney, configs, teams)
    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        do_tournament_one_epoch_async(channels, tourney, teams, configs)
    end
end

function run_tournament(tourney, configs)
    teams = [Symbol(t) for t in configs["TEAMS"]]
    winners = init_winners(teams)
    for k=1:tourney.epochs
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs)
        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end
    end
end

"""
    run_state_space_tournament(configs)

Run a tournament parameterized by `configs` which keeps track of the exploration of state space and mutations.
"""
function run_state_space_tournament(configs)
    tourney = Tournament(configs)
    @info "Starting tournament $(tourney.unique_id)"
    tournament_path = get_tournament_path(configs, tourney.unique_id)

    teams = [Symbol(t) for t in configs["TEAMS"]]
    team_to_perturb = Dict{Symbol, DecisionModel}()
    team_to_public_perturb = Dict{Symbol, DecisionModel}()
    markov_teams = [t for t in teams if get_player_config(configs, "TYPE", t) == "HybridPlayer"]
    winners = init_winners(teams)::Dict{Union{Symbol, Nothing}, Int}
    state_to_value = StateValueContainer(configs)

    for k=1:tourney.epochs
        @info "epoch $k / $(tourney.epochs)"
        @info "$state_to_value"

        # Add a new perturbation to player's model weights
        initialize_epoch!(configs, team_to_perturb, team_to_public_perturb, markov_teams, tourney, k)

        with_enrichment = conf -> create_enriched_players(conf, state_to_value, team_to_perturb, team_to_public_perturb)
        epoch_winners = do_tournament_one_epoch(tourney, teams, configs; create_players = with_enrichment)
        println(epoch_winners)

        for (w,n) in collect(epoch_winners)
            winners[w] += n
        end

        epoch_winner = epoch_winners[1][1]
        validation_check = validate_mutation!(configs, state_to_value, team_to_perturb, team_to_public_perturb, markov_teams, epoch_winner)
        
        if validation_check
            # Part of validation check ensures epoch_winner is not `nothing`
            apply_mutation!(team_to_perturb, team_to_public_perturb, markov_teams, epoch_winner::Symbol)
        end

        finalize_epoch!(team_to_perturb, team_to_public_perturb, configs, tournament_path, k, markov_teams, epoch_winner, validation_check)
    end
    println(winners)
end

function get_tournament_path(configs, tourney_id)
    models_dir = get_player_config(configs, "MODELS_DIR")
    tournament_path = joinpath(models_dir, "tournament_$(tourney_id)")
    ~isdir(tournament_path) && mkdir(tournament_path)
    return tournament_path
end

function initialize_epoch!(configs::Dict, team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel}, teams, tourney, epoch_num)
    if epoch_num == 1
        for team in teams
            # First iteration, start with stored model
            team_to_perturb[team] = try_load_serialized_model(team, configs)::DecisionModel
            team_to_public_perturb[team] = try_load_serialized_public_model(team, configs)::DecisionModel
        end
    end

    # Linear spacing of reward weight across epochs 
    value_weight = epoch_num/(tourney.epochs-1)
    @info "setting value weight to $value_weight"

    for team in teams
        if get_player_config(configs, "MODIFY_REINFORCEMENT_WEIGHTS", team)
            Catan.set_player_config(configs, team, "VALUE_WEIGHT", value_weight)
            Catan.set_player_config(configs, team, "REWARD_WEIGHT", 1 - value_weight)
        end
        # Every other iteration, start with perturbed model
        add_perturbation!(team_to_perturb[team], 0.1)
        add_perturbation!(team_to_public_perturb[team], 0.1)
    end
end

function finalize_epoch!(team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel}, configs, tourney_path, epoch_num, teams, winner, validation_check::Bool)
    if validation_check
        for team in teams
            write_perturbed_linear_model(tourney_path, epoch_num, team, team_to_perturb[team], get_player_config(configs, "MODELS_DIR", team))
            write_perturbed_linear_model(tourney_path, epoch_num, team, team_to_public_perturb[team], get_player_config(configs, "MODELS_DIR", team), "public_model")
        end
    end
end

function validate_mutation!(configs::Dict, state_to_value::StateValueContainer, team_to_perturb::Dict, team_to_public_perturb::Dict, teams::AbstractVector{Symbol}, winner::Union{Nothing, Symbol})::Bool
    @info "Starting validation epoch for winner $winner"
    if winner === nothing || !(winner in teams)
        @info "skipping mutation since player $winner won epoch"
        return false
    end
    perturb = team_to_perturb[winner]
    public_perturb = team_to_public_perturb[winner]
    # TODO validate
    # Create 1 hybrid players with the mutation and compare to baseline model
    validation_configs = deepcopy(configs)
    tourney = Tournament(configs)

    validation_team_to_perturb = Dict{Symbol, DecisionModel}([winner => perturb])
    validation_team_to_public_perturb = Dict{Symbol, DecisionModel}([winner => public_perturb])

    with_enrichment = conf -> create_enriched_players(conf, state_to_value, validation_team_to_perturb, validation_team_to_public_perturb)
    ordered_winners = do_tournament_one_epoch(tourney, teams, validation_configs; create_players = with_enrichment)::Vector{Tuple{Union{Symbol, Nothing}, Int}}
    validation_check = ordered_winners[1][1] == winner
    if validation_check
        @info "Finished validation epoch: accept mutation"
    else
        @warn "Finished validation epoch: reject mutation $(ordered_winners[1])"
    end
    return validation_check
end

function apply_mutation!(team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel}, teams::AbstractVector{Symbol}, winner::Symbol)

    @info "$winner won the epoch, so all players copy his mutation"
    for team in teams
        team_to_perturb[team].weights = copy(team_to_perturb[winner].weights)
        team_to_public_perturb[team].weights = copy(team_to_public_perturb[winner].weights)
    end

end

function create_enriched_players(configs, state_to_value::StateValueContainer, team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel})
    players = Catan.create_players(configs)

    # Enrich players if needed
    for p in players
        if typeof(p) <: MarkovPlayer
            p.process.state_to_value = state_to_value.master
            p.process.new_state_to_value = state_to_value.current
            p.model = get(team_to_perturb, p.player.team, p.model)
            p.model_public = get(team_to_public_perturb, p.player.team, p.model_public)
        end
    end
    return players
end
