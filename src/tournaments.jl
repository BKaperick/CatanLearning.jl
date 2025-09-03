function initialize_tournament(configs::Dict)
    if configs["WRITE_FEATURES"] == true
        @info "Initializing player feature files"
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
    teams = [Symbol(t) for t in configs["TEAMS"]]
    winners = init_winners(teams)
    return teams,winners
end

function get_tournament_path(configs, tourney_id)
    models_dir = joinpath(@__DIR__, get_player_config(configs, "MODELS_DIR"))
    tournament_path = joinpath(models_dir, "tournament_$(tourney_id)")
    ~isdir(models_dir) && mkdir(models_dir)
    ~isdir(tournament_path) && mkdir(tournament_path)
    return tournament_path
end
function TournamentConfig(tournament_configs::Dict, player_configs::Dict)
    tourney_id = generate_tournament_id()
    return TournamentConfig(tournament_configs["GAMES_PER_MAP"], tournament_configs["MAPS_PER_EPOCH"], tournament_configs["NUM_EPOCHS"], 
    tournament_configs["GENERATE_RANDOM_MAPS"],
    tourney_id, get_tournament_path(player_configs, tourney_id))
end

function Tournament(configs::Dict)
    teams,winners = initialize_tournament(configs::Dict)
    Tournament(TournamentConfig(configs["Tournament"], configs), teams, winners)
end
function AsyncTournament(configs::Dict)
    teams,_ = initialize_tournament(configs::Dict)
    AsyncTournament(TournamentConfig(configs["Tournament"], configs), teams, Catan.read_channels_from_config(configs))
end
function MutatingTournament(configs::Dict)
    teams,winners = initialize_tournament(configs::Dict)
    MutatingTournament(TournamentConfig(configs["Tournament"], configs), teams, winners, :rule)
end

#
# ONE TOURNAMENT
#

function run_tournament(configs::Dict)
    tourney = Tournament(configs)
    run(tourney, configs)
    @info tourney.winners
    return tourney.winners
end

"""
    run_state_space_tournament(configs)

Run a tournament parameterized by `configs` which keeps track of the exploration of state space and mutations.
Each epoch adds a new mutation
"""
function run_state_space_tournament(configs)
    tourney = MutatingTournament(configs)
    run(tourney, configs)
end


function run(tourney::AbstractTournament, configs::Dict)
    data_points = 4*(tourney.configs.games_per_map * tourney.configs.maps_per_epoch * tourney.configs.epochs)
    @info "Running tournament of $(data_points/4) games in total"

    for k=1:tourney.configs.epochs
        @info "epoch $k / $(tourney.configs.epochs)"
        initialize_epoch!(tourney)
        do_tournament_one_epoch(tourney, configs)
        finalize_epoch!(tourney)
    end
end

function run_async(tourney::AsyncTournament, configs::Dict)
    @sync begin
        @async run(tourney, configs)
        @async consume_feature_channel!(tourney.channels[:main], data_points, configs["PlayerSettings"]["FEATURES"])
        @async consume_feature_channel!(tourney.channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"])
    end
    #=_run_tournament_async(tourney, configs)
    consume_feature_channel!(tourney.channels[:main], data_points, configs["PlayerSettings"]["FEATURES"])
    consume_feature_channel!(tourney.channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"])
    =#
end

function run(tourney::MutatingTournament, configs::Dict)
    @info "Starting tournament $(tourney.unique_id)"

    team_to_perturb = Dict{Symbol, DecisionModel}()
    team_to_public_perturb = Dict{Symbol, DecisionModel}()
    markov_teams = [t for t in tourney.teams if get_player_config(configs, "TYPE", t) == "HybridPlayer"]
    
    for k=1:tourney.configs.epochs
        @info "epoch $k / $(tourney.configs.epochs)"

        # Add a new perturbation to player's model weights
        initialize_epoch!(tourney, configs, team_to_perturb, team_to_public_perturb, markov_teams, k)
        prev_winners = copy(tourney.winners)
        with_enrichment = conf -> create_enriched_players(conf, team_to_perturb, team_to_public_perturb)
        do_tournament_one_epoch(tourney, configs; create_players = with_enrichment)
        epoch_winners = order_winners(merge(-, tourney.winners, prev_winners))
        println(epoch_winners)

        epoch_winner = epoch_winners[1][1]
        validation_check = validate_mutation!(configs, team_to_perturb, team_to_public_perturb, markov_teams, epoch_winner)
        
        if validation_check
            # Part of validation check ensures epoch_winner is not `nothing`
            apply_mutation!(team_to_perturb, team_to_public_perturb, markov_teams, epoch_winner::Symbol)
        end

        finalize_epoch!(team_to_perturb, team_to_public_perturb, configs, tourney.configs.path, k, markov_teams, epoch_winner, validation_check)
    end
    println(tourney.winners)
end

#
# ONE EPOCH
#

function initialize_epoch!(tourney::Union{Tournament, AsyncTournament})
    #tourney.winners = init_winners(tourney.teams)
end

"""
    initialize_epoch!(configs::Dict, team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel}, teams, tourney, epoch_num)

The start of a new epoch for a mutating tournament (see `run_state_space_tournament`) requires several steps before running the epoch games:
    * Modify Value and Reward weighting for MarkovPlayers
    * Add perturbation to LearningPlayer DecisionModels
"""
function initialize_epoch!(tourney::MutatingTournament, configs::Dict, team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel}, teams, epoch_num)
    if epoch_num == 1
        for team in teams
            # First iteration, start with stored model
            team_to_perturb[team] = try_load_serialized_model(team, configs)::DecisionModel
            team_to_public_perturb[team] = try_load_serialized_public_model(team, configs)::DecisionModel
        end
    end

    # Linear spacing of reward weight across epochs 
    value_weight = epoch_num/(tourney.configs.epochs-1)

    for team in teams
        if get_player_config(configs, "MODIFY_REINFORCEMENT_WEIGHTS", team)
            @info "setting value weight for $team to $value_weight"
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


function finalize_epoch!(tourney::Tournament)
    @info tourney.winners
end

function finalize_epoch!(tourney::AsyncTournament)
end

function do_tournament_one_epoch(tourney::AbstractTournament, configs; create_players = Catan.create_players)
    map_str = ""
    if !tourney.configs.generate_random_maps
        map_str = read(configs["LOAD_MAP"], String)
    end
    for j=1:tourney.configs.maps_per_epoch
        @info "map $j / $(tourney.configs.maps_per_epoch)"
        if tourney.configs.generate_random_maps
            do_tournament_one_map!(tourney, configs, j; create_players = create_players)
        else
            @debug "loading map from $(configs["LOAD_MAP"])"
            do_tournament_one_map!(tourney, configs, j, map_str; create_players = create_players)
        end
    end
end

#
# ONE MAP
#

function do_tournament_one_map!(tourney::AbstractTournament, configs, map_num::Integer; create_players = Catan.create_players)
    map = Catan.generate_random_map()
    do_tournament_one_map!(tourney, configs, map_num, map; create_players)
end

function do_tournament_one_map!(tourney::Union{MutatingTournament, Tournament}, configs, map_num::Integer, map_str::AbstractString; create_players = Catan.create_players)
    
    function log_games_per_map(map_num, tourney, i)
        g_num = (map_num - 1)*tourney.configs.games_per_map + i
        if g_num % 1 == 0
            @debug "Game $(g_num) / $(tourney.configs.maps_per_epoch * tourney.configs.games_per_map)"
        end
    end

    iter_logger = (tourney, i) -> log_games_per_map(map_num, tourney, i)
    do_tournament_one_map!(tourney, configs, map_str, iter_logger; create_players)
end

function do_tournament_one_map!(tourney, configs, map_str::AbstractString, iter_logger; create_players = Catan.create_players)
    map = Map(map_str)
    for i=1:tourney.configs.games_per_map
        main_logger = descend_logger(configs, "GAME")
            players = create_players(configs)
        winner = do_tournament_one_game!(map, players, configs)
        tourney.winners[winner] += 1
        global_logger(main_logger)
        iter_logger(tourney, i)
    end
end

function do_tournament_one_map!(tourney::AsyncTournament, configs, map_num::Integer, map_str::AbstractString; create_players = Catan.create_players)
    @debug "running do_tournament_one_map_async!"
    map = Map(map_str)
    for i=1:tourney.configs.games_per_map
        players = create_players(configs)
        do_tournament_one_game_async!(tourney.channels, map, players, configs)
        #yield()
    end
end

#
# ONE GAME
#

function do_tournament_one_game!(map::Map, players, configs)
    game = Game(players, configs)
    board = Board(map, configs)
    _,winner = Catan.run(game, board)
    @debug "finished game $(game.unique_id)"

    w = winner
    if winner !== nothing
        w = winner.player.team
    end

    return w
end

function do_tournament_one_game_async!(channels, map::Map, players, configs)
    game = Game(players, configs)
    board = Board(map, configs)
    @debug "starting game $(game.unique_id)"
    main_logger = global_logger()
    global_logger(NullLogger())
    _,winner = Catan.run_async(channels, game, board)
    global_logger(main_logger)
    @debug "finished game $(game.unique_id)"
    return
end

#
# HELPER METHODS
#

function init_winners(teams)::Dict{Union{Symbol, Nothing}, Int}
    winners = Dict{Union{Symbol, Nothing}, Int}([(k,0) for k in teams])
    winners[nothing] = 0
    return winners
end

function consume_feature_channel!(channel, count, key)
    _write_many_features_file(channel, count, key)
    close(channel)
end


function validate_mutation!(configs::Dict, team_to_perturb::Dict, team_to_public_perturb::Dict, teams::AbstractVector{Symbol}, winner::Union{Nothing, Symbol})::Bool
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

    with_enrichment = conf -> create_enriched_players(conf, validation_team_to_perturb, validation_team_to_public_perturb)
    ordered_winners = do_tournament_one_epoch(tourney, validation_configs; create_players = with_enrichment)::Vector{Tuple{Union{Symbol, Nothing}, Int}}
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
        if team == winner
            continue
        end
        team_to_perturb[team].weights = copy(team_to_perturb[winner].weights)
        team_to_public_perturb[team].weights = copy(team_to_public_perturb[winner].weights)
    end

end

function create_enriched_players(configs, team_to_perturb::Dict{Symbol, DecisionModel}, team_to_public_perturb::Dict{Symbol, DecisionModel})
    players = Catan.create_players(configs)

    # Enrich players if needed
    for p in players
        if typeof(p) <: MarkovPlayer
            p.model = get(team_to_perturb, p.player.team, p.model)
            p.model_public = get(team_to_public_perturb, p.player.team, p.model_public)
        end
    end
    return players
end
