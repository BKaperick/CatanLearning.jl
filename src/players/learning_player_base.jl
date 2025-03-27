include("../learning/feature_computation.jl")
include("../learning/production_model.jl")
using Catan: GameApi, BoardApi, PlayerApi, random_sample_resources, get_random_resource,
             construct_city, construct_settlement, construct_road,
             do_play_devcard, propose_trade_goods, do_robber_move_theft,
            get_admissible_theft_victims, choose_road_location

using Catan: choose_next_action, choose_place_robber, do_post_action_step

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


action_types = Dict([
    :ConstructSettlement => :Deterministic,
    :ConstructCity => :Stochastic,
    :ConstructCity => :OtherPlayersInput
])


function get_legal_action_sets(board::Board, players::Vector{PlayerPublicView}, player::Player, actions::Set{Symbol})::Vector{AbstractActionSet}
    main_action_set = ActionSet(:Deterministic)
    action_sets = Vector{AbstractActionSet}([main_action_set])
    
    if :ConstructCity in actions
        candidates = BoardApi.get_admissible_city_locations(board, player.team)
        for coord in candidates
            push!(main_action_set.actions, Action(:ConstructCity, (g, b, p) -> construct_city(b, p.player, coord), coord))
        end
    end
    if :ConstructSettlement in actions
        candidates = BoardApi.get_admissible_settlement_locations(board, player.team)
        for coord in candidates
            push!(main_action_set.actions, Action(:ConstructSettlement, (g, b, p) -> construct_settlement(b, p.player, coord), coord))
        end
    end
    if :ConstructRoad in actions
        candidates = BoardApi.get_admissible_road_locations(board, player.team)
        for coord in candidates
            push!(main_action_set.actions, Action(:ConstructRoad, (g, b, p) -> construct_road(b, p.player, coord[1], coord[2]), coord[1], coord[2]))
        end
    end

    if :BuyDevCard in actions
        action_set = ActionSet{SampledAction}(:BuyDevCard)
        estimated_remaining_devcards = get_estimated_remaining_devcards(board, players, player)
        sampled_devcards = Catan.random_sample_resources(estimated_remaining_devcards, 5, true)
        for card in sampled_devcards 
            push!(action_set.actions, SampledAction(:BuyDevCard, 
                                             (g, b, p) -> deterministic_draw_devcard(g, p, card),
                                             (g, b, p) -> Catan.draw_devcard(g, p.player)))
        end
        push!(action_sets, action_set)
    end

    if :PlayDevCard in actions
        devcards = PlayerApi.get_admissible_devcards(player)
        for (card,cnt) in devcards
            # TODO how do we stop them playing devcards first turn they get them?  Is this correctly handled in get_admissible call?
            if card != :VictoryPoint
                push!(main_action_set.actions, Action(:PlayDevCard, (g, b, p) -> do_play_devcard(b, g.players, p, card), card))
            end
        end
    end
    
    # TODO: this is leaking info from other players, since `propose_trade_goods` 
    # asks the other user if they would accept the offered trade, so the player 
    # can check if the trade would be accepted before deciding to do it.
    if :ProposeTrade in actions
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

    if :PlaceRobber in actions
        # Get candidates
        for candidate_tile = BoardApi.get_admissible_robber_tiles(board)
            candidate_victims = get_admissible_theft_victims(board, players, player, candidate_tile)
            for victim in candidate_victims
                
                # Here, we have one ActionSet per set of parameters
                action_set = ActionSet{SampledAction}(:PlaceRobber)
                resources = get_estimated_resources(board, players, victim)
                for r in resources
                    push!(action_set.actions, 
                          SampledAction(
                                 Symbol("$(r)"), 
                                 (g, b, p) -> do_robber_move_theft(
                                                                   b, g.players, 
                                                                   p, victim, 
                                                                   candidate_tile, 
                                                                   resource), 
                                 (g, b, p) -> do_robber_move_theft(b, p, victim, candidate_tile),
                                 victim, candidate_tile))
                end
                push!(action_sets, action_set)
            end
        end
    end

    return action_sets
end

function Catan.choose_road_location(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Vector{Tuple{Int, Int}}})::Union{Nothing,Vector{Tuple{Int, Int}}}
    x = Catan.choose_next_action(board, players, player, Set([:ConstructRoad]))
    #actions = Set([:ConstructRoad])
    #x = Catan.choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})
    (args, action) = x
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
function get_action_with_features(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, actions::Set{Symbol})::Action
    action_sets = get_legal_action_sets(board, players, player.player, actions)
    best_actions = ActionSet(:SecondRound)
    analyze_actions!(board, players, player, action_sets)
    for (i,set) in enumerate(action_sets)
        @assert all([a.win_proba != nothing for a in set.actions])
        push!(best_actions.actions, aggregate(set))
    end
    return aggregate(best_actions)
end

function analyze_actions!(board::Board, players::Vector{PlayerPublicView}, player::PlayerType, action_sets::Vector{AbstractActionSet})
    for (i,set) in enumerate(action_sets)
        println("Analyzing set: $(typeof(set)) :: $(set.name)")
        for action in set.actions
            analyze_action!(action, board, players, player)
        end
    end
end

"""
    `aggregate(set::ActionSet)`

Identifies the best parameters to use for this action
"""
function aggregate(set::ActionSet)::Action
    println("Aggregating $(set.name) with features $([(a.name, a.win_proba) for a in set.actions])")
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
function analyze_action!(action::AbstractAction, board::Board, players::Vector{PlayerPublicView}, player::PlayerType)
    hypoth_board = deepcopy(board)
    hypoth_player = deepcopy(player)
    hypoth_game = Game([DefaultRobotPlayer(p.team) for p in players])
    action.func!(hypoth_game, hypoth_board, hypoth_player)
    action.features = compute_features(hypoth_board, hypoth_player.player)
    
    # TODO Temporal difference algo does this later, so we don't want to double compute
    action.win_proba = predict_model(player.machine, action.features)
    return action
end

"""
    `choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})`

Gathers all legal actions, and chooses the one that most increases the player's 
probability of victory, based on his `player.machine` model.  If no action 
increases the probability of victory, then do nothing.
"""
function Catan.choose_next_action(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, actions::Set{Symbol})::Tuple
    print("choosing action")
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
    
    #best_action, reachable_states = get_action_with_features(board, players, player, actions)
    best_action = get_action_with_features(board, players, player, actions)
    
    # TODO we could even make the "no-op" action and use that as a possibility?
    # Only do an action if it will improve his estimated chances of winning
    if best_action.win_proba > current_win_proba
        @info "And his chance of winning will go to $(best_action.proba) with this next move"
        
        return (best_action.args, best_action.func!)
    end
    return (nothing, nothing)
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
