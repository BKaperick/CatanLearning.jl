include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
import Catan: choose_next_action, choose_place_robber
using Catan: GameApi, BoardApi, PlayerApi, random_sample_resources, get_random_resource,
             construct_city, construct_settlement, construct_road,
             do_play_devcard, propose_trade_goods, do_robber_move_theft

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

function get_legal_action_functions(board::Board, players::Vector{PlayerPublicView}, player::Player, actions::Set{Symbol})
    action_functions = []
    
    if :ConstructCity in actions
        candidates = BoardApi.get_admissible_city_locations(board, player.team)
        for coord in candidates
            push!(action_functions, (:ConstructCity, (g, b, p) -> construct_city(b, p.player, coord)))
        end
    end
    if :ConstructSettlement in actions
        candidates = BoardApi.get_admissible_settlement_locations(board, player.team)
        for coord in candidates
            push!(action_functions, (:ConstructSettlement, (g, b, p) -> construct_settlement(b, p.player, coord)))
        end
    end
    if :ConstructRoad in actions
        candidates = BoardApi.get_admissible_road_locations(board, player.team)
        for coord in candidates
            push!(action_functions, (:ConstructRoad, (g, b, p) -> construct_road(b, p.player, coord[1], coord[2])))
        end
    end

    if :BuyDevCard in actions
        estimated_remaining_devcards = get_estimated_remaining_devcards(board, players, player)
        sampled_devcards = Catan.random_sample_resources(estimated_remaining_devcards, 5, true)
        for card in sampled_devcards 
            push!(action_functions, (:BuyDevCard, (g, b, p) -> deterministic_draw_devcard(g, p, card)))
        end
    end

    if :PlayDevCard in actions
        devcards = PlayerApi.get_admissible_devcards(player)
        for (card,cnt) in devcards
            # TODO how do we stop them playing devcards first turn they get them?  Is this correctly handled in get_admissible call?
            if card != :VictoryPoint
                push!(action_functions, (:PlayDevCard, (g, b, p) -> do_play_devcard(b, g.players, p, card)))
            end
        end
    end
    
    # TODO: this is leaking info from other players, since `propose_trade_goods` 
    # asks the other user if they would accept the offered trade, so the player 
    # can check if the trade would be accepted before deciding to do it.
    if :ProposeTrade in actions
        sampled = random_sample_resources(player.resources, 1)
        rand_resource_from = [sampled...]
        rand_resource_to = [get_random_resource()]
        while rand_resource_to[1] == rand_resource_from[1]
            rand_resource_to = [get_random_resource()]
        end
        push!(action_functions, (:ProposeTrade, (g, b, p) -> propose_trade_goods(b, g.players, p, rand_resource_from, rand_resource_to)))
    end

    if :PlaceRobber in actions
        # Get candidates
        for new_tile = BoardApi.get_admissible_robber_tiles(board)
            # TODO stochastic
            push!(action_functions, (:PlaceRobber, (g, b, p) -> do_robber_move_theft(b, g.players, p, new_tile)))
        end
    end

    return action_functions
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

action_types = Dict([
    :ConstructSettlement => :Deterministic,
    :ConstructCity => :Stochastic,
    :ConstructCity => :OtherPlayersInput
])

"""
    `get_action_with_features(board::Board, players::Vector{PlayerPublicView}, 
                              player::PlayerType, actions::Set{Symbol})`

Gets the legal action functions for the player at this board state, and 
computes the feature vector for each resulting state.  This is a critical 
helper function for all the machine-learning players.
"""
function get_action_with_features(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set{Symbol})
    action_functions = get_legal_action_functions(board, players, player.player, actions)
    reachable_states = Dict([(t, []) for t in Set([x[1] for x in action_functions])])
    for (i,(action_type, action_func!)) in enumerate(action_functions)
        hypoth_board = deepcopy(board)
        hypoth_player = deepcopy(player)
        hypoth_game = Game([DefaultRobotPlayer(p.team) for p in players])
        action_func!(hypoth_game, hypoth_board, hypoth_player)
        features = compute_features(hypoth_board, hypoth_player.player)

        push!(reachable_states[action_type], (action_func!, features))
    end
    return reachable_states
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
    #machine = ml_machine(player)
    machine = player.machine
    #current_features = compute_features(board, inner_player(player))
    current_features = compute_features(board, player.player)
    current_win_proba = predict_model(machine, current_features)
    # @info "$(inner_player(player).team) thinks his chance of winning is $(current_win_proba)"
    
    type_to_actions_and_features = get_action_with_features(board, players, player, actions)
    for (i,action_type) in enumerate(collect(keys(type_to_actions_and_features)))
        probas = []
        for (action_func!, features) in type_to_actions_and_features[action_type]
            push!(probas, predict_model(machine, features))
        end
        p = sum(probas) / length(probas) # usually 1

        if p > best_action_proba
            best_action_proba = p
            best_action_type = action_type
        end
    end

    # Only do an action if it will improve his estimated chances of winning
    if best_action_proba > current_win_proba
        @info "And his chance of winning will go to $(best_action_proba) with this next move"
        if best_action_type == :BuyDevCard
            return (g, b, p) -> draw_devcard(g, p) 
        end
        return actions_and_features[best_action_type][1]
    end
    return nothing
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
