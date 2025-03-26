include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
import Catan: choose_next_action, choose_place_robber
using Catan: GameApi, BoardApi, PlayerApi, random_sample_resources, get_random_resource,
             construct_city, construct_settlement, construct_road,
             do_play_devcard, propose_trade_goods, do_robber_move_theft,
            get_admissible_theft_victims, choose_road_location


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


#abstract type AbstractAction
mutable struct Action
    args::Tuple
    name::Symbol
    func!::Function
    win_proba::Union{nothing, Float64}
    features::Vector
end

mutable struct ActionSet
    name::Symbol
    actions::Vector{Action}
end

function Action(name::Symbol, func::Function, args...) 
    println("Adding action $args ($(typeof(args)))")
    Action(args, func!, name, nothing)
end

function aggregate(action_set::ActionSet)
    for action in action_set.actions
        if action
    end
end

action_types = Dict([
    :ConstructSettlement => :Deterministic,
    :ConstructCity => :Stochastic,
    :ConstructCity => :OtherPlayersInput
])


function get_legal_action_sets(board::Board, players::Vector{PlayerPublicView}, player::Player, actions::Set{Symbol})::Vector{ActionSet}
    action_sets = []
    
    if :ConstructCity in actions
        action_set = ActionSet(:ConstructCity)
        candidates = BoardApi.get_admissible_city_locations(board, player.team)
        for coord in candidates
            push!(action_set.actions, Action(:ConstructCity, (g, b, p) -> construct_city(b, p.player, coord), coord))
        end
        push!(action_sets, action_set)
    end
    if :ConstructSettlement in actions
        action_set = ActionSet(:ConstructSettlement)
        candidates = BoardApi.get_admissible_settlement_locations(board, player.team)
        for coord in candidates
            push!(action_set.actions, Action(:ConstructSettlement, (g, b, p) -> construct_settlement(b, p.player, coord), coord))
        end
        push!(action_sets, action_set)
    end
    if :ConstructRoad in actions
        action_set = ActionSet(:ConstructRoad)
        candidates = BoardApi.get_admissible_road_locations(board, player.team)
        for coord in candidates
            push!(action_set.actions, Action(:ConstructRoad, (g, b, p) -> construct_road(b, p.player, coord[1], coord[2]), coord[1], coord[2]))
        end
        push!(action_sets, action_set)
    end

    if :BuyDevCard in actions
        action_set = ActionSet(:BuyDevCard)
        estimated_remaining_devcards = get_estimated_remaining_devcards(board, players, player)
        sampled_devcards = Catan.random_sample_resources(estimated_remaining_devcards, 5, true)
        for card in sampled_devcards 
            push!(action_set.actions, Action(:BuyDevCard, (g, b, p) -> deterministic_draw_devcard(g, p, card)))
        end
        push!(action_sets, action_set)
    end

    if :PlayDevCard in actions
        action_set = ActionSet(:BuyDevCard)
        devcards = PlayerApi.get_admissible_devcards(player)
        for (card,cnt) in devcards
            # TODO how do we stop them playing devcards first turn they get them?  Is this correctly handled in get_admissible call?
            if card != :VictoryPoint
                push!(action_set.actions, Action(:PlayDevCard, (g, b, p) -> do_play_devcard(b, g.players, p, card), card))
            end
        end
        push!(action_sets, action_set)
    end
    
    # TODO: this is leaking info from other players, since `propose_trade_goods` 
    # asks the other user if they would accept the offered trade, so the player 
    # can check if the trade would be accepted before deciding to do it.
    if :ProposeTrade in actions
        action_set = ActionSet(:ProposeTrade)
        sampled = random_sample_resources(player.resources, 1)
        rand_resource_from = [sampled...]
        rand_resource_to = [get_random_resource()]
        while rand_resource_to[1] == rand_resource_from[1]
            rand_resource_to = [get_random_resource()]
        end
        push!(action_set.actions, 
              Action(
                     :ProposeTrade, 
                     (g, b, p) -> propose_trade_goods(
                                                      b, g.players, p, 
                                                      rand_resource_from, 
                                                      rand_resource_to), 
                     rand_resource_from, rand_resource_to))
        push!(action_sets, action_set)
    end

    if :PlaceRobber in actions
        action_set = ActionSet(:PlaceRobber)
        # Get candidates
        for candidate_tile = BoardApi.get_admissible_robber_tiles(board)
            candidate_victims = get_admissible_theft_victims(board, players, player, candidate_tile)
            for victim in candidate_victims
                resources = get_estimated_resources(board, players, victim)
                for r in resources
                    push!(action_set.actions, 
                          Action(
                                 Symbol("PlaceRobber_$(candidate_tile)_$(victim.team)"), 
                                 (g, b, p) -> do_robber_move_theft(
                                                                   b, g.players, 
                                                                   p, victim, 
                                                                   candidate_tile, 
                                                                   resource), 
                                 victim, candidate_tile))
                end
            end
        end
        push!(action_sets, action_set)
    end

    return action_sets
end

function Catan.choose_road_location(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Vector{Tuple{Int, Int}}})::Union{Nothing,Vector{Tuple{Int, Int}}}
    args, action = choose_next_action(board, players, player, Set([:ConstructRoad]))
    println("chose road: $args")
    return args
end

function deterministic_do_robber_move_theft(board, player, victim, new_robber_tile, resource)

do_robber_move_theft(board, player::PlayerType, victim::Player, new_robber_tile::Symbol, stolen_good::Symbol)
    do_robber_move_theft(board, player, victim, new_robber_tile)
end

"""
    `deterministic_draw_devcard(game, player, card)`

Equivalent to `Catan.draw_devcard`, but we pass an explicit card choice, since we're sampling from our estimated
card counts, rather than leaking info from the main one during the hypothetical games.
"""
function deterministic_draw_devcard(game, player, card)
    GameApi._draw_devcard(game, card)
    PlayerApi.buy_devcard(player.player, card)
end

"""
    `get_action_with_features(board::Board, players::Vector{PlayerPublicView}, 
                              player::PlayerType, actions::Set{Symbol})`

Gets the legal action functions for the player at this board state, and 
computes the feature vector for each resulting state.  This is a critical 
helper function for all the machine-learning players.
"""
function get_action_with_features(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set{Symbol})
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    types = Set([a.name for a in action_sets])
    reachable_states = Dict([(t, []) for t in types])
    for (i,set) in enumerate(action_sets)
        for action in set.actions
            analyze_action!(action, board, players, player)
            push!(reachable_states[action.name], action)
        end
    end
    return reachable_states
end

function analyze_action(action::Action, board::Board, players::Vector{PlayerPublicView}, player::PlayerType)
    hypoth_board = deepcopy(board)
    hypoth_player = deepcopy(player)
    hypoth_game = Game([DefaultRobotPlayer(p.team) for p in players])
    action.func!(hypoth_game, hypoth_board, hypoth_player)
    action.features = compute_features(hypoth_board, hypoth_player.player)
    action.win_proba = predict_model(player.machine, action.features)
    return action
end

"""
    `choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})`

Gathers all legal actions, and chooses the one that most increases the player's 
probability of victory, based on his `player.machine` model.  If no action 
increases the probability of victory, then do nothing.
"""
function choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})
    best_action_type = nothing
    best_action_proba = -1
    best_action_index = 1
    best_args = nothing
    #machine = ml_machine(player)
    machine = player.machine
    #current_features = compute_features(board, inner_player(player))
    current_features = compute_features(board, player.player)
    current_win_proba = predict_model(machine, current_features)
    # @info "$(inner_player(player).team) thinks his chance of winning is $(current_win_proba)"
    
    type_to_actions_and_features = get_action_with_features(board, players, player, actions)
    for (i,action_type) in enumerate(collect(keys(type_to_actions_and_features)))
        probas = []
        for (args, action_func!, features) in type_to_actions_and_features[action_type]
            push!(probas, predict_model(machine, features))
        end
        if action_type == :BuyDevCard || 
            (action_type != nothing && contains(String(action_type), "PlaceRobber"))
            p = sum(probas) / length(probas)
        else
            p = maximum(probas)
        end

        if p > best_action_proba
            best_action_proba = p
            best_action_type = action_type
            best_action_index = argmax(probas)
            best_args = type_to_actions_and_features[action_type][1]
        end
    end

    # Only do an action if it will improve his estimated chances of winning
    if best_action_proba > current_win_proba
        @info "And his chance of winning will go to $(best_action_proba) with this next move"

        # TODO cleanup this hack-y exception for BuyDevCard and PlaceRobber
        if best_action_type == :BuyDevCard
            return (g, b, p) -> draw_devcard(g, p) 

        elseif contains(String(best_action_type), "PlaceRobber")
            (_, victim_team, tile) = split(String(best_action_type), "_")
            return (g, b, p) -> do_robber_move_theft(b, p, [pp for pp in g.players 
                                           if pp.player.team == victim_team][1], 
                                    tile)
        else
            return best_args, type_to_actions_and_features[best_action_type][best_action_index]
        end
    end
    return nothing, nothing
end

function save_parameters_after_game_end(file::IO, board::Board, players::Vector{PlayerType}, player::PlayerType, winner_team::Union{Nothing, Symbol})
    features = compute_features(board, player.player)
    # For now, we just use a binary label to say who won
    label = get_csv_friendly(player.player.team == winner_team)
    values = join([get_csv_friendly(f[2]) for f in features], ",")
    write(file, "$values,$label\n")
end

# TODO implement this based on ML model, only accept trade if win proba augments more than the other player's win proba from the trade
# function choose_accept_trade(board::Board, player::RobotPlayer, from_player::PlayerPublicView, from_goods::Vector{Symbol}, to_goods::Vector{Symbol})::Bool
