using Catan: GameApi, BoardApi, PlayerApi, PreAction, ChosenAction, random_sample_resources, unsafe_random_sample_one_resource, get_random_resource,
             construct_city, construct_settlement, construct_road,
             do_play_devcard, propose_trade_goods, do_robber_move_theft,
            get_admissible_theft_victims, choose_road_location, trade_goods, choose_building_location

using Catan: choose_next_action, choose_who_to_trade_with,
             choose_place_robber, do_post_action_step, 
             choose_accept_trade, choose_resource_to_draw,
             choose_one_resource_to_discard,
             choose_robber_victim, inner_do_robber_move_theft,
             get_state_score

function get_estimated_resources(board::Board, players::AbstractVector{PlayerPublicView}, target::PlayerPublicView)::Dict{Symbol, Int8}
    resources = random_sample_resources(Catan.RESOURCE_TO_COUNT, target.resource_count)
    return Dict(counter(resources))
end
"""
    `get_estimated_remaining_devcards`

A helper function for the learning player to make a probabilistic decision 
about what remains in the devcard deck based on public information.
"""
function get_estimated_remaining_devcards(board::Board, players::AbstractVector{PlayerPublicView}, player::Player)::Dict{Symbol, Int}
    devcards = Catan.get_devcard_counts(board.configs)
    for (card,count) in collect(player.devcards)
        devcards[card] -= count
    end
    for (card,count) in collect(player.devcards_used)
        devcards[card] -= count
    end
    for other_player in players
        for (card,count) in collect(other_player.devcards_used)
            devcards[card] -= count
        end
    end
    return devcards
end

function get_legal_action_sets(board::Board, players::AbstractVector{PlayerPublicView}, player::Player, pre_actions::Set{PreAction})::Vector{AbstractActionSet}
    main_action_set = ActionSet{Action}(:Deterministic)
    action_sets = Vector{AbstractActionSet}([])

    actions = Dict([(p.name, p.admissible_args) for p in pre_actions])
    
    @debug "$player can do one of following: $(join(actions, "\n"))"

    # Deterministic PreActions are all quite easy to handle
    for (action, candidates) in actions

        # This is because `PreAction` currently doesn't have any way to represent an 
        # action passing in candidates, and *then* sampling
        if action == :PlaceRobber
            continue
        end

        for args in candidates
            func! = nothing
            if action == :ConstructCity
                func! = (g, b, p) -> construct_city(b, p.player, args...)
            elseif action == :ConstructSettlement
                func! = (g, b, p) -> construct_settlement(b, p.player, args...)
            elseif action == :ConstructRoad
                func! = (g, b, p) -> construct_road(b, p.player, args...)
            elseif action == :PlayDevCard
                func! = (g, b, p) -> do_play_devcard(b, g.players, p, args...)
            elseif action == :GainResource
                func! = (g, b, p) -> Catan.harvest_one_resource!(b, p.player, args..., 1)
            elseif action == :LoseResource
                func! = (g, b, p) -> Catan.PlayerApi.take_resource!(p.player, args...)
            elseif action == :AcceptTrade
                func! = (g, b, p) -> Catan.trade_goods(args[1], p.player, args[2:end]...)
            
            else
                @assert false "Found unexpected action $action while handling deterministic actions"
            end
            push!(main_action_set.actions, Action(action, func!, args...))
        end
    end

    if haskey(actions, :DoNothing)
        push!(main_action_set.actions, Action(:DoNothing, Returns(nothing), ()))
    end
    if haskey(actions, :BuyDevCard)
        action_set = ActionSet{SampledAction}(:BuyDevCard)
        estimated_remaining_devcards = get_estimated_remaining_devcards(board, players, player)
        sampled_devcards = Catan.random_sample_resources(estimated_remaining_devcards, 5, true)
        for card in sampled_devcards 
            push!(action_set.actions, SampledAction(:BuyDevCard, 
                                             (g, b, p) -> deterministic_draw_devcard(g, b, p, card)))
        end
        if length(action_set.actions) > 0
            push!(action_sets, action_set)
        end
    end
    
    # TODO: this is leaking info from other players, since `propose_trade_goods` 
    # asks the other user if they would accept the offered trade, so the player 
    # can check if the trade would be accepted before deciding to do it.
    if haskey(actions, :ProposeTrade)
        #action_set = ActionSet{Action}(:ProposeTrade)
        #sampled = random_sample_resources(player.resources, 1)
        #rand_resource_from = [sampled...]
        rand_resource_from = [unsafe_random_sample_one_resource(player.resources)]
        rand_resource_to = get_random_resource()
        while rand_resource_to == rand_resource_from[1]
            rand_resource_to = get_random_resource()
        end
        to_goods_dict = Dict{Symbol, UInt8}([rand_resource_to => UInt8(1)])
        to_goods = [rand_resource_to]
        push!(main_action_set.actions, 
              Action(
                     :ProposeTrade, 
                     (g, b, p) -> propose_trade_goods(
                                                      b, g.players, p, 
                                                      rand_resource_from, 
                                                      to_goods, to_goods_dict), 
                     rand_resource_from, to_goods))
    end

    if haskey(actions, :PlaceRobber)
        candidates = actions[:PlaceRobber]
        num_isolated_robber_actions = UInt8(1)
        # Get candidates
        for candidate_tiles = candidates
            candidate_tile = candidate_tiles[1]
            candidate_victims = get_admissible_theft_victims(board, players, player, candidate_tile)
            for victim in candidate_victims
                victim_team = victim.team    
                # Here, we have one ActionSet per set of parameters
                action_set = ActionSet{SampledAction}(:PlaceRobber)
                resources = get_estimated_resources(board, players, victim)
                for r in keys(resources)
                    push!(action_set.actions, 
                          SampledAction(
                                 Symbol("PlaceRobber_$(r)"), 
                                 (g, b, p) -> inner_do_robber_move_theft(
                                                                   b, g.players, 
                                                                   p, victim.team, 
                                                                   candidate_tile, 
                                                                   r), 
                                 victim, candidate_tile))
                end
                push!(action_sets, action_set)
            end
            if length(candidate_victims) == 0 && num_isolated_robber_actions > 0
                push!(main_action_set.actions, 
                      Action(:PlaceRobber, 
                             (g, b, p) -> inner_do_robber_move_theft(b, g.players, p, nothing, candidate_tile, nothing), 
                             nothing, candidate_tile))
                num_isolated_robber_actions -= UInt8(1)
            end
        end
    end
    
    if length(main_action_set.actions) > 0
        push!(action_sets, main_action_set)
    end

    for set in action_sets
        @assert length(set.actions) > 0 "$(set.name) is empty"
    end
    
    return action_sets
end

"""
    `get_best_transition(board::Board, players::AbstractVector{PlayerPublicView}, 
                              player::PlayerType, actions::Set{Symbol})`

Gets the legal action functions for the player at this board state, and 
computes the feature vector for each resulting state.  This is a critical 
helper function for all the machine-learning players.
"""
function get_best_transition(board::Board, players::AbstractVector{PlayerPublicView}, player::PlayerType, actions::Set, depth::UInt8=UInt8(1))::MarkovTransition
    do_current_state_calculation(player, board, players)
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    return analyze_and_aggregate_action_sets(board, players, player, action_sets, depth)
end

function do_current_state_calculation(player::LearningPlayer, board::Board, players::AbstractVector{PlayerPublicView})::Nothing
end

function do_current_state_calculation(player::MarkovPlayer, board::Board, players::AbstractVector{PlayerPublicView})::Nothing
    features = compute_features(board, player.player)
    player.current_state = MarkovState(player.process, features, player.model)
    return
end

function analyze_and_aggregate_action_sets(board::Board, players::AbstractVector{PlayerPublicView}, player::PlayerType, action_sets::Vector{AbstractActionSet}, depth::UInt8)::MarkovTransition
    # Converts the inner actions to MarkovTransitions with `reward` and `features` properties
    transitions = analyze_actions!(board, players, player, action_sets, depth)
    return aggregate(transitions)
end

function analyze_actions!(board::Board, players::AbstractVector{PlayerPublicView}, player::PlayerType, action_sets::Vector{AbstractActionSet}, depth::UInt8)::Vector{MarkovTransition}
    transitions = Vector{MarkovTransition}([])
    for set in action_sets
        @debug "analyzing $set"
        
        states = Vector{MarkovState}([])
        for action in set.actions
            state = analyze_action!(action, board, players, player, depth)
            push!(states, state)
        end

        if set isa ActionSet{SampledAction}
            update_transitions!(transitions, set::ActionSet{SampledAction}, states)
        else
            update_transitions!(transitions, set::ActionSet{Action}, states)
        end
    end
    return transitions
end

function update_transitions!(transitions, set::ActionSet{SampledAction}, states)
    transition = MarkovTransition(states, set.actions[1])
    @debug "sampled: $transition"
    push!(transitions, transition)
end

function update_transitions!(transitions, set::ActionSet{Action}, states)
    for (state,action) in zip(states, set.actions)
        transition = MarkovTransition([state], action)
        @debug "$(state.key): $transition"
        push!(transitions, transition)
    end
end

function aggregate(ts::Vector{MarkovTransition})::MarkovTransition
    return argmax(t -> t.reward, ts)
end

function isolate_and_perform_action(action::AbstractAction, hypoth_game::Game, hypoth_board::Board, hypoth_player::PlayerType)
    # We control the log-level of 'hypothetical' games separately from the main game.
    
    @debug "Entering game $(hypoth_game.unique_id) with action $(action) and log level $(hypoth_game.configs["LogSettings"]["HYPOTH_LOG_LEVEL"])"
    main_logger = descend_logger(hypoth_player.player.configs, "HYPOTH")
    main_save_lvl = hypoth_game.configs["SAVE_GAME_TO_FILE"]
    hypoth_game.configs["SAVE_GAME_TO_FILE"] = false
    
    action.func!(hypoth_game, hypoth_board, hypoth_player)

    hypoth_game.configs["SAVE_GAME_TO_FILE"] = main_save_lvl
    global_logger(main_logger)
    @debug "Reverting to global logger as we exit HYPOTH environment $(hypoth_game.unique_id)"
end

function calculate_state_score(hypoth_game::Game, hypoth_board::Board, players::AbstractVector{PlayerPublicView}, hypoth_player::LearningPlayer, depth::UInt8)
    # Don't apply first turn calculations here
    # Not exactly the correct condition, since technically first turn happens a few times (2 settlements, 2 roads)
    hypoth_game.turn_num = 2
    next_legal_actions = Catan.get_legal_actions(hypoth_game, hypoth_board, hypoth_player.player)
    filtered_next_legal_actions = Set{PreAction}()
    for a in next_legal_actions
        push!(filtered_next_legal_actions, a)
    end
    return get_best_transition(hypoth_board, players, hypoth_player, filtered_next_legal_actions, depth + UInt8(1)).reward
end

function Catan.get_state_score(board::Board, player::LearningPlayer)::Float64
    features = compute_features(board, player.player)
    predict_model(player.model, features)
end
function Catan.get_state_score(player::LearningPlayer, features::Vector{Pair{Symbol, Float64}})::Float64
    predict_model(player.model, features)
end

function create_hypoth_other_players(board::Board, players::AbstractVector{PlayerPublicView})::AbstractVector{PlayerType}
    out = Vector{PlayerType}(undef, length(players))
    for (i,p) in enumerate(players)
        r = get_estimated_resources(board, players, p)
        player = DefaultRobotPlayer(p.team, board.configs, r)
        out[i] = player
    end
    return out
end

function analyze_action!(action::AbstractAction, board::Board, players::AbstractVector{PlayerPublicView}, player::PlayerType, depth::UInt8)::MarkovState
    hypoth_board = copy(board)
    hypoth_player = copy(player)
    hypoth_game = Game(create_hypoth_other_players(board, players), board.configs)
    
    isolate_and_perform_action(action, hypoth_game, hypoth_board, hypoth_player)
    
    features = compute_features(hypoth_board, hypoth_player.player)
    
    # Look ahead an additional `SEARCH_DEPTH` turns
    if depth < get_player_config(hypoth_player, "SEARCH_DEPTH")
        state_score = calculate_state_score(hypoth_game, hypoth_board, players, hypoth_player, depth)
    else

        state_score = get_state_score(hypoth_player, features)
    end
    return MarkovState(features, state_score)
end

"""    
    `choose_next_action(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})`

Gathers all legal actions, and chooses the one that most increases the player's 
probability of victory, based on his `player.model` model.  If no action 
increases the probability of victory, then do nothing.
"""
function Catan.choose_next_action(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, actions::Set{PreAction})::ChosenAction
    @debug "$(player) considers $(collect(actions))"
    best_action = get_best_transition(board, players, player, actions).chosen_action
    @info "$(player) chooses to $(best_action)"
    return best_action #ChosenAction(best_action.name, best_action.args...)
end

function Catan.choose_road_location(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Tuple{Tuple{TInt, TInt}, Tuple{TInt, TInt}}}, do_pay_cost::Bool = true)::Union{Nothing,Tuple{Tuple{TInt, TInt}, Tuple{TInt, TInt}}} where {TInt <: Integer}
    args = Vector{Tuple}([(c...,do_pay_cost) for c in candidates])
    best_action = get_best_transition(board, players, player, Set([PreAction(:ConstructRoad, args)]))
    return Tuple(best_action.chosen_action.args[1:2])
end

function Catan.choose_building_location(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Tuple{TInt, TInt}}, building_type::Symbol, is_first_turn::Bool = false)::Union{Nothing,Tuple{TInt,TInt}} where {TInt <: Integer}
    @debug "learning player has $candidates as choices to build"
    pre_action_name = building_type == :City ? :ConstructCity : :ConstructSettlement
    pre_actions = Set([PreAction(pre_action_name, Vector{Tuple}([(c,is_first_turn) for c in candidates]))])
    action = get_best_transition(board, players, player, pre_actions)
    return action.chosen_action.args[1]
end

function Catan.choose_place_robber(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, candidate_tiles::Vector{Symbol})::Symbol
    return get_best_transition(board, players, player, Set([PreAction(:PlaceRobber, candidate_tiles)])).chosen_action.args[2]
end

function Catan.choose_resource_to_draw(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer)::Symbol
    resources = collect(keys(Dict((k,v) for (k,v) in board.resources if v > 0)))
    return get_best_transition(board, players, player, Set([PreAction(:GainResource, resources)])).chosen_action.args[1]
end

function Catan.choose_one_resource_to_discard(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer)::Symbol
    isempty(player.player.resources) && throw(ArgumentError("Player has no resources"))
    resources = [r for (r,v) in player.player.resources if v > 0]
    pre_actions = Set([PreAction(:LoseResource, resources)])
    action = get_best_transition(board, players, player, pre_actions)
    return action.chosen_action.args[1]
end

"""
    `deterministic_draw_devcard(game, board, player, card)`

Equivalent to `Catan.draw_devcard`, but we pass an explicit card choice, since we're sampling from our estimated
card counts, rather than leaking info from the main one during the hypothetical games.
"""
function deterministic_draw_devcard(game, board, player, card)
    GameApi._draw_devcard(game, card)
    BoardApi.pay_construction!(board, :DevelopmentCard)
    PlayerApi.buy_devcard(player.player, card)
end

function Catan.choose_accept_trade(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer, from_player::Player, from_goods::Vector{Symbol}, to_goods::Vector{Symbol})::Bool
    actions = Set([PreAction(:DoNothing), PreAction(:AcceptTrade, [(from_player, from_goods, to_goods)])])
    best_action = get_best_transition(board, players, player, actions)
    return best_action.chosen_action.args !== nothing
end

"""
    `Catan.choose_who_to_trade_with(board::Board, player::LearningPlayer, players::AbstractVector{PlayerPublicView})::Symbol`

Use public model (stored in `player.player.model_public`) to choose a trading partner as the weakest among the options
"""
function Catan.choose_who_to_trade_with(board::Board, player::LearningPlayer, players::AbstractVector{PlayerPublicView})::Symbol
    return argmin(p -> predict_public_model(player.model_public, board, p), players).team
end

"""
    choose_robber_victim(board::Board, player::RobotPlayer, 
    potential_victims::PlayerPublicView...)::PlayerPublicView

Use public model (stored in `player.player.model_public`) to choose a trading partner as the strongest among the options
"""
function Catan.choose_robber_victim(board::Board, player::LearningPlayer, potential_victims::PlayerPublicView...)::PlayerPublicView
    return argmax(p -> predict_public_model(player.model_public, board, p), potential_victims)
    @info "$(player) decided it is wisest to steal from the $(max_ind) player"
    return max_ind
end

"""
    choose_monopoly_resource(board::Board, players::AbstractVector{PlayerPublicView}, 
    player::RobotPlayer)::Symbol

Called during the Monopoly development card action.  Choose the resource to steal from each player based on public information.
"""
function choose_monopoly_resource(board::Board, players::AbstractVector{PlayerPublicView}, player::LearningPlayer)::Symbol
    #= TODO do something smarter with resource estimates
    for player in players
        get_estimated_resources(board, players, target)::Dict{Symbol, Int}
    end
    =#
    return get_best_transition(board, players, player, Set(PreAction(:GainResource, [(r,) for r in Catan.RESOURCES]))).chosen_action.args[1]
end
