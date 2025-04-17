#include("../learning/feature_computation.jl")
#include("../learning/production_model.jl")
using Catan: GameApi, BoardApi, PlayerApi, PreAction, random_sample_resources, get_random_resource,
             construct_city, construct_settlement, construct_road,
             do_play_devcard, propose_trade_goods, do_robber_move_theft,
            get_admissible_theft_victims, choose_road_location, trade_goods, choose_building_location

using Catan: choose_next_action, choose_who_to_trade_with,
             choose_place_robber, do_post_action_step, 
             choose_accept_trade, choose_resource_to_draw,
             choose_one_resource_to_discard

function get_estimated_resources(board::Board, players::Vector{PlayerPublicView}, target::PlayerPublicView)::Dict{Symbol, Int}
    return Dict([(r,1) for r in Catan.RESOURCES])
end
"""
    `get_estimated_remaining_devcards`

A helper function for the learning player to make a probabilistic decision 
about what remains in the devcard deck based on public information.
"""
function get_estimated_remaining_devcards(board::Board, players::Vector{PlayerPublicView}, player::Player)::Dict{Symbol, Int}
    devcards = deepcopy(Catan.DEVCARD_COUNTS)
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

function get_legal_action_sets(board::Board, players::Vector{PlayerPublicView}, player::Player, pre_actions::Set{PreAction})::Vector{AbstractActionSet}

    main_action_set = ActionSet(:Deterministic)
    action_sets = Vector{AbstractActionSet}([])

    actions = Dict([(p.name, p.admissible_args) for p in pre_actions])
    
    # Deterministic PreActions are all quite easy to handle
    for (action, candidates) in actions
        for args in candidates
            func! = nothing
            if action == :ConstructCity
                func! = (g, b, p) -> construct_city(b, p.player, args)
            elseif action == :ConstructSettlement
                func! = (g, b, p) -> construct_settlement(b, p.player, args)
            elseif action == :ConstructRoad
                func! = (g, b, p) -> construct_road(b, p.player, args...)
            elseif action == :PlayDevCard
                func! = (g, b, p) -> do_play_devcard(b, g.players, p, args)
            elseif action == :GainResource
                func! = (g, b, p) -> Catan.harvest_one_resource!(b, p.player, args, 1)
            elseif action == :LoseResource
                func! = (g, b, p) -> Catan.PlayerApi.take_resource!(p.player, args)
            elseif action == :AcceptTrade
                func! = (g, b, p) -> Catan.trade_goods(args[1], p.player, args[2:end]...)
            #elseif action == :DoNothing
            #    func! = (g,b,p) -> ()
            
            # This is because `PreAction` currently doesn't have any way to represent an 
            # action passing in candidates, and *then* sampling
            elseif action == :PlaceRobber
                break
            else
                @assert false "Found unexpected action $action while handling deterministic actions"
            end
            push!(main_action_set.actions, Action(action, func!, args))
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
                                             (g, b, p) -> deterministic_draw_devcard(g, b, p, card),
                                             (g, b, p) -> Catan.draw_devcard(g, p.player)))
        end
        push!(action_sets, action_set)
    end
    
    # TODO: this is leaking info from other players, since `propose_trade_goods` 
    # asks the other user if they would accept the offered trade, so the player 
    # can check if the trade would be accepted before deciding to do it.
    if haskey(actions, :ProposeTrade)
        #action_set = ActionSet{Action}(:ProposeTrade)
        sampled = random_sample_resources(player.resources, 1)
        rand_resource_from = [sampled...]
        rand_resource_to = [get_random_resource()]
        while rand_resource_to[1] == rand_resource_from[1]
            rand_resource_to = [get_random_resource()]
        end
        push!(main_action_set.actions, 
              Action(
                     :ProposeTrade, 
                     (g, b, p) -> propose_trade_goods(
                                                      b, g.players, p, 
                                                      rand_resource_from, 
                                                      rand_resource_to), 
                     rand_resource_from, rand_resource_to))
        #push!(action_sets, action_set)
    end

    if haskey(actions, :PlaceRobber)
        candidates = actions[:PlaceRobber]
        # Get candidates
        for candidate_tile = candidates
            candidate_victims = get_admissible_theft_victims(board, players, player, candidate_tile)
            for victim in candidate_victims
                victim_team = victim.team    
                # Here, we have one ActionSet per set of parameters
                action_set = ActionSet{SampledAction}(:PlaceRobber)
                resources = get_estimated_resources(board, players, victim)
                for r in keys(resources)
                    push!(action_set.actions, 
                          SampledAction(
                                 Symbol("$(r)"), 
                                 (g, b, p) -> do_robber_move_theft(
                                                                   b, g.players, 
                                                                   p, victim.team, 
                                                                   candidate_tile, 
                                                                   r), 
                                 (g, b, p) -> do_robber_move_theft(b, p, victim.team, candidate_tile),
                                 victim, candidate_tile))
                end
                push!(action_sets, action_set)
            end
            if length(candidate_victims) == 0
                push!(main_action_set.actions, 
                      Action(:PlaceRobber, 
                             (g, b, p) -> do_robber_move_theft(b, g.players, p, nothing, candidate_tile, nothing), 
                             nothing, candidate_tile))
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

function Catan.choose_road_location(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Vector{Tuple{Int, Int}}})::Union{Nothing,Vector{Tuple{Int, Int}}}
    cand_args = Set([Tuple(t) for t in candidates])
    best_action = get_best_action(board, players, player, Set([PreAction(:ConstructRoad, cand_args)]))
    return collect(best_action.args[1])
end

function Catan.choose_building_location(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Tuple{Int, Int}}, building_type::Symbol)::Union{Nothing,Tuple{Int,Int}}
    @debug "learning player has $candidates as choices to build"
    if building_type == :City
        return get_best_action(board, players, player, Set([PreAction(:ConstructCity, candidates)])).args[1]
    else
        return get_best_action(board, players, player, Set([PreAction(:ConstructSettlement, candidates)])).args[1]
    end
end

function Catan.choose_place_robber(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidate_tiles::Vector{Symbol})::Symbol
    return get_best_action(board, players, player, Set([PreAction(:PlaceRobber, candidate_tiles)])).args[2]
end

function Catan.choose_resource_to_draw(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer)::Symbol
    resources = collect(keys(Dict((k,v) for (k,v) in board.resources if v > 0)))
    return get_best_action(board, players, player, Set([PreAction(:GainResource, resources)])).args[1]
end

function Catan.choose_one_resource_to_discard(board::Board, player::LearningPlayer)::Symbol
    resources = [r for (r,v) in player.player.resources if v > 0]
    return get_best_action(board, Vector{PlayerPublicView}([]), player, Set([PreAction(:LoseResource, resources)])).args[1]
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

"""
    `get_best_action(board::Board, players::Vector{PlayerPublicView}, 
                              player::PlayerType, actions::Set{Symbol})`

Gets the legal action functions for the player at this board state, and 
computes the feature vector for each resulting state.  This is a critical 
helper function for all the machine-learning players.
"""
function get_best_action(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set, depth::Int=1)::Action
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    return analyze_and_aggregate_action_sets(board, players, player, action_sets, depth)
end

function analyze_and_aggregate_action_sets(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, action_sets::Vector{AbstractActionSet}, depth::Int)::Action
    best_actions = ActionSet(:SecondRound)

    # Enriches the inner actions with `win_proba` and `features` properties
    analyze_actions!(board, players, player, action_sets, depth)
    for (i,set) in enumerate(action_sets)
        # Aggregate chooses the best action from each set, and pushes it into the best_actions set
        push!(best_actions.actions, aggregate(set))
    end
    @debug "$(player.player.team) is now choosing among $(join([a.name for a in best_actions.actions], ", "))"

    # Aggregate chooses the best action from the `best_actions` set
    return aggregate(best_actions)
end

function analyze_actions!(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, action_sets::Vector{AbstractActionSet}, depth::Int)
    for (i,set) in enumerate(action_sets)
        @debug "analyzing action set ($(length(set.actions)) actions): \n$(join(["$(a.name)($(a.args))" for a in set.actions], "\n"))"
        for action in set.actions
            analyze_action!(action, board, players, player, depth)
        end
    end
end

"""
    `aggregate(set::ActionSet)`

Identifies the best parameters to use for this action
"""
function aggregate(set::ActionSet)::Action
    return argmax(a -> a.win_proba, set.actions)
end
function aggregate(set::ActionSet{SampledAction})::Action
    avg_proba = sum([a.win_proba for a in set.actions]) / length(set.actions)
    # an ActionSet{SampledAction} contains only actions with the same func (they differ only in Sampling Func)
    # TODO some way to enforce this in the code?
    func! = set.actions[1].func!
    args = set.actions[1].args
    return Action(set.name, avg_proba, func!, args)
end


function analyze_action!(action::AbstractAction, board::Board, players::Vector{PlayerPublicView}, player::PlayerType, depth::Int)
    hypoth_board = deepcopy(board)
    hypoth_player = deepcopy(player)
    hypoth_game = Game([DefaultRobotPlayer(p.team, board.configs) for p in players], board.configs)
    @debug "Entering hypoth game $(hypoth_game.unique_id) with action $(action.name)($(action.args))"
    main_logger = global_logger()
    global_logger(NullLogger())
    action.func!(hypoth_game, hypoth_board, hypoth_player)
    action.features = compute_features(hypoth_board, hypoth_player.player)
    global_logger(main_logger)
    
    @debug "Leaving hypoth game $(hypoth_game.unique_id)"
    
    # Look ahead an additional `MAX_DEPTH` turns
    
    
    if depth < get_player_config(player, "SEARCH_DEPTH")
        next_legal_actions = Catan.get_legal_actions(hypoth_game, hypoth_board, hypoth_player.player)
        action.win_proba = get_best_action(hypoth_board, players, hypoth_player, next_legal_actions, depth + 1).win_proba
        @debug "after performing $(action.name), there are $(length(next_legal_actions)) possibilities"
    else
        # TODO Temporal difference algo does this later, so we don't want to double compute
        action.win_proba = predict_model(player.machine, action.features)
    end


    return action
end

"""    
    `choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})`

Gathers all legal actions, and chooses the one that most increases the player's 
probability of victory, based on his `player.machine` model.  If no action 
increases the probability of victory, then do nothing.
"""
function Catan.choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{PreAction})::Tuple
    @info "$(player.player.team) considers $(collect(actions))"
    best_action = get_best_action(board, players, player, actions)
    @info "$(player.player.team) chooses to $(best_action.name) $(best_action.args)"
    return (best_action.args, best_action.func!)
end

function Catan.choose_accept_trade(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, from_player::Player, from_goods::Vector{Symbol}, to_goods::Vector{Symbol})::Bool
    actions = Set([PreAction(:DoNothing), PreAction(:AcceptTrade, (from_player, from_goods, to_goods))])
    best_action = get_best_action(board, players, player, actions)
    return best_action.args !== nothing
end

"""
    `Catan.choose_who_to_trade_with(board::Board, player::LearningPlayer, players::Vector{PlayerPublicView})::Symbol`

Use public model (stored in `player.player.machine_public`) to choose a trading partner as the weakest among the options
"""
function Catan.choose_who_to_trade_with(board::Board, player::LearningPlayer, players::Vector{PlayerPublicView})::Symbol
    return argmin(p -> predict_public_model(player.machine_public, board, p), players).team
end
#function Catan.choose_monopoly_resource(board::Board, players::Vector{PlayerPublicView}, player::RobotPlayer)::Symbol
#end

