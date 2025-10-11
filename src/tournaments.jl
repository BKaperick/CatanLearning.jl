function initialize_tournament!(configs::Dict)
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
end

function get_tournament_path(configs, tourney_id)
    models_dir = get_player_config(configs, "MODELS_DIR")
    tournament_path = joinpath(models_dir, "tournament_$(tourney_id)")
    @info "Making directory $(models_dir)"
    ~isdir(models_dir) && mkdir(models_dir)
    ~isdir(tournament_path) && mkdir(tournament_path)
    return tournament_path
end
function TournamentConfig(tournament_configs::Dict, player_configs::Dict)
    tourney_id = generate_tournament_id()
    players = Catan.create_players(player_configs)
    return TournamentConfig(players, tournament_configs["GAMES_PER_MAP"], tournament_configs["MAPS_PER_EPOCH"], tournament_configs["NUM_EPOCHS"], 
    tournament_configs["GENERATE_RANDOM_MAPS"],
    tourney_id, get_tournament_path(player_configs, tourney_id))
end

function Tournament(configs::Dict)
    initialize_tournament!(configs::Dict)
    Tournament(TournamentConfig(configs["Tournament"], configs), Dict(), Catan.read_channels_from_config(configs))
end
function AsyncTournament(configs::Dict)
    initialize_tournament!(configs::Dict)
    AsyncTournament(TournamentConfig(configs["Tournament"], configs), Catan.read_channels_from_config(configs))
end
function MutatingTournament(configs::Dict)
    initialize_tournament!(configs::Dict)
    MutatingTournament(TournamentConfig(configs["Tournament"], configs), Dict())
end

get_markov_players(tourney::AbstractTournament) = Channel() do c
    for player in tourney.configs.players
        if player isa MarkovPlayer
            push!(c, player)
        end
    end
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

function run(tourney::T, configs::Dict)::T where T <: AbstractTournament
    data_points = 4*(tourney.configs.games_per_map * tourney.configs.maps_per_epoch * tourney.configs.epochs)
    @info "Running tournament $(tourney.configs.unique_id) with $(data_points/4) games in total"
    _run(tourney, configs)
    return tourney
end

function _run(tourney::Union{Tournament, MutatingTournament}, configs::Dict)
    _run_tournament(tourney, configs)
    return tourney
end

function _run(tourney::AsyncTournament, configs::Dict)
    data_points = 4*(tourney.configs.games_per_map * tourney.configs.maps_per_epoch * tourney.configs.epochs)
    @sync begin
        errormonitor(Threads.@spawn _run_tournament(tourney, configs))
        errormonitor(Threads.@spawn consume_feature_channel!(configs, tourney.channels[:main], data_points, configs["PlayerSettings"]["FEATURES"]))
        errormonitor(Threads.@spawn consume_feature_channel!(configs, tourney.channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"]))
    end
end

function initialize_players!(tourney::AbstractTournament)

    # Ensure all markov players are sharing the same StateValueContainer
    svc = nothing
    for (i,player) in enumerate(get_markov_players(tourney))
        if i == 1
            svc = player.process.state_values
        else
            player.process.state_values = svc
        end
    end
end

function _run_tournament(tourney::AbstractTournament, configs::Dict)
    initialize_players!(tourney)
    for k=1:tourney.configs.epochs
        @info "epoch $k / $(tourney.configs.epochs)"
        
        initialize_epoch!(tourney, configs, k)
        do_tournament_one_epoch(tourney, configs)
        finalize_epoch!(tourney, configs, k)
    end
    finalize_tournament(tourney, configs)
end

function finalize_tournament(tourney::MutatingTournament, configs)
    # If the tournament didn't use its directory, then remove it
    if length(readdir(tourney.configs.path)) == 0
        rm(tourney.configs.path)
    end
end

function finalize_tournament(tourney::Union{Tournament, AsyncTournament}, configs)
    close(tourney.channels[:main])
    close(tourney.channels[:public])
end
#
# ONE EPOCH
#

function initialize_epoch!(tourney::AbstractTournament, configs::Dict, epoch_num)
end

"""
    initialize_epoch!(tourney::MutatingTournament, configs::Dict, epoch_num)

The start of a new epoch for a mutating tournament (see `run_state_space_tournament`) requires several steps before running the epoch games:
    * Modify Value and Reward weighting for MarkovPlayers
    * Add perturbation to LearningPlayer DecisionModels
"""
function initialize_epoch!(tourney::MutatingTournament, configs::Dict, epoch_num)
    # Linear spacing of reward weight across epochs 
    value_weight = epoch_num/(tourney.configs.epochs-1)
    empty!(tourney.winners)

    for player in get_markov_players(tourney)
        team = player.player.team
        if get_player_config(configs, "MODIFY_REINFORCEMENT_WEIGHTS", team)
            @info "setting value weight for $team to $value_weight"
            Catan.set_player_config(configs, team, "VALUE_WEIGHT", value_weight)
            Catan.set_player_config(configs, team, "REWARD_WEIGHT", 1 - value_weight)
        end
        # Every other iteration, start with perturbed model
        add_perturbation!(player.model, 0.1)
        add_perturbation!(player.model_public, 0.1)
    end
end

function finalize_epoch!(tourney::MutatingTournament, configs, epoch_num)

    epoch_winners = order_winners(tourney.winners)
    println(epoch_winners)

    epoch_winner = epoch_winners[1][1]
    validation_check = validate_mutation!(configs, epoch_winner)
    
    if validation_check
        # Part of validation check ensures epoch_winner is not `nothing`
        apply_mutation!(get_markov_players(tourney), epoch_winner::Symbol)
    end
    
    if validation_check
        for player in get_markov_players(tourney)
            team = player.player.team
            write_perturbed_linear_model(tourney.configs.path, epoch_num, team, player.model, get_player_config(configs, "MODELS_DIR", team))
            write_perturbed_linear_model(tourney.configs.path, epoch_num, team, player.model_public, get_player_config(configs, "MODELS_DIR", team), "public_model")
        end
    end
end


function finalize_epoch!(tourney, _, __)
    @info tourney.winners
end

function finalize_epoch!(tourney::AsyncTournament, _, __)
end

function do_tournament_one_epoch(tourney::AbstractTournament, configs::Dict)
    map_str = ""
    if !tourney.configs.generate_random_maps
        map_str = read(configs["LOAD_MAP"], String)
    end
    for j=1:tourney.configs.maps_per_epoch
        @info "map $j / $(tourney.configs.maps_per_epoch)"
        #@warn "T: $(repr(UInt64(pointer_from_objref(tourney.configs.players[1].player))))"
        do_tournament_one_map!(tourney, tourney.configs.players, configs, j, map_str)
    end
end

function do_tournament_one_epoch(tourney::Tournament, configs::Dict)
    map_str = ""
    if !tourney.configs.generate_random_maps
        map_str = read(configs["LOAD_MAP"], String)
    end

    Threads.@threads for j=1:tourney.configs.maps_per_epoch
        # We can't share players across threads
        thread_players = SVector{length(tourney.configs.players), PlayerType}([copy(p) for p in tourney.configs.players])

        @debug "Thread $(Threads.threadid()): $(repr(UInt64(pointer_from_objref(thread_players[1].player))))"
        @info "map $j / $(tourney.configs.maps_per_epoch)"
        do_tournament_one_map!(tourney, thread_players, configs, j, map_str)

        data_points = 4*(tourney.configs.games_per_map)

        errormonitor(Threads.@spawn consume_feature_channel!(configs, tourney.channels[:main], data_points, configs["PlayerSettings"]["FEATURES"]))
        errormonitor(Threads.@spawn consume_feature_channel!(configs, tourney.channels[:public], data_points, configs["PlayerSettings"]["PUBLIC_FEATURES"]))
    end
end

#
# ONE MAP
#

function do_tournament_one_map!(tourney::AbstractTournament, players::AbstractVector{PlayerType}, configs::Dict, map_num::Integer, map_str::AbstractString)
    iter_logger = (tourney, i) -> log_games_per_map(map_num, tourney.configs, i)
    if tourney.configs.generate_random_maps
        map = Map(Catan.generate_random_map())
    else
        map = Map(map_str)
    end

    for i=1:tourney.configs.games_per_map
        @debug "game $i / $(tourney.configs.games_per_map)"
        main_logger = descend_logger(configs, "GAME")

        Catan.refresh_players!(players)
        @assert all(PlayerApi.is_in_initial_state.([p.player for p in players]))
        
        do_tournament_one_game!(tourney, map, players, configs)

        global_logger(main_logger)
        iter_logger(tourney, i)
    end
end

#
# ONE GAME
#

function do_tournament_one_game!(tourney::Union{Tournament, MutatingTournament}, map::Map, players, configs)
    game = Game(players, configs)
    board = Board(map, configs)
    _,winner = Catan.run(game, board)
    @debug "finished game $(game.unique_id)"

    w = winner
    if winner !== nothing
        w = winner.player.team
    end
    if haskey(tourney.winners, w)
        tourney.winners[w] += 1
    else
        tourney.winners[w] = 1
    end
    return
end

function do_tournament_one_game!(tourney::AsyncTournament, map::Map, players, configs)
    game = Game(players, configs)
    board = Board(map, configs)
    Catan.run_async(tourney.channels, game, board)
    @debug "finished game $(game.unique_id)"
    return
end

function do_tournament_one_game!(tourney::Tournament, map::Map, players, configs)
    game = Game(players, configs)
    board = Board(map, configs)
    board, winner = Catan.run_async(tourney.channels, game, board)
    @debug "finished game $(game.unique_id)"
    
    w = winner
    if winner !== nothing
        w = winner.player.team
    end
    if haskey(tourney.winners, w)
        tourney.winners[w] += 1
    else
        tourney.winners[w] = 1
    end
    return
end

#
# HELPER METHODS
#

function consume_feature_channel!(configs, channel, count, key)
    if configs["WRITE_FEATURES"] == true
        _write_many_features_file(channel, count, key)
    end
end

function validate_mutation!(configs::Dict, winner::Union{Nothing, Symbol})::Bool
    @info "Starting validation epoch for winner $winner"
    if winner === nothing
        @info "skipping mutation since player $winner won epoch"
        return false
    end
    # TODO validate
    # Create 1 hybrid players with the mutation and compare to baseline model
    validation_configs = deepcopy(configs)
    # TODO should we pass the existing players in here as well ?
    tourney = Tournament(configs)

    do_tournament_one_epoch(tourney, validation_configs)
    ordered_winners = order_winners(tourney.winners)
    validation_check = ordered_winners[1][1] == winner

    if validation_check
        @info "Finished validation epoch: accept mutation"
    else
        @warn "Finished validation epoch: reject mutation $(ordered_winners[1])"
    end
    return validation_check
end

function apply_mutation!(markov_players::AbstractVector{PlayerType}, winner_team::Symbol)
    winners = [p for p in markov_players if p.player.team == winner_team]
    
    if length(winners) == 0
        @info "Skipping mutation application since winner $(winner_team) is not a MarkovPlayer"
        return
    end
    
    winner = winners[1]

    @info "$winner won the epoch, so all players copy his mutation"
    for player in markov_players
        if player.player.team == winner
            continue
        end
        player.model.weights = copy(winner.model.weights)
        player.model_public.weights = copy(winner.model_public.weights)
    end
end
    
function log_games_per_map(map_num, tourney_configs, i)
    g_num = (map_num - 1)*tourney_configs.games_per_map + i
    if g_num % 1 == 0
        @debug "Game $(g_num) / $(tourney_configs.maps_per_epoch * tourney_configs.games_per_map)"
    end
end
