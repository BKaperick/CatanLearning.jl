using Revise
using Test
using TestItems
using TestItemRunner
using StatsBase
using JET
using Logging

using Catan
using CatanLearning
using Catan: Game, Board, Player, PlayerType, PlayerApi, BoardApi, GameApi,
             DefaultRobotPlayer, PlayerPublicView, 
             ALL_ACTIONS, choose_accept_trade, get_legal_actions,
load_gamestate!,
reset_savefile!,
test_player_implementation,
setup_players,
setup_and_do_robot_game,
test_automated_game,
parse_configs,
doset

using CatanLearning:
    compute_features,
    EmpathRobotPlayer,
    HybridPlayer,

read_values_file,
feature_library,
get_state_optimizing_quantity,
predict_model,
MarkovState,
MaxValueMarkovPolicy,
MaxRewardMarkovPolicy,
TemporalDifferencePlayer,
get_legal_action_sets,
feature_library


@testsnippet global_test_setup begin
    using JET
    using StatsBase

    using Catan
    using CatanLearning
    using Catan: Game, Board, Player, PlayerType, PlayerApi, BoardApi, GameApi,
                DefaultRobotPlayer, PlayerPublicView, 
                ALL_ACTIONS, choose_accept_trade, get_legal_actions,
    load_gamestate!,
    reset_savefile!,
    test_player_implementation,
    setup_players,
    setup_and_do_robot_game,
    test_automated_game,
    parse_configs,
    doset

    using CatanLearning:
        compute_features,
        EmpathRobotPlayer,
        HybridPlayer,

    read_values_file,
    feature_library,
    get_state_optimizing_quantity,
    predict_model,
    MarkovState,
    MaxValueMarkovPolicy,
    MaxRewardMarkovPolicy,
    TemporalDifferencePlayer,
    get_legal_action_sets,
    feature_library

    configs = parse_configs("Configuration.toml")


features = collect(keys(feature_library))

features_increasing_good = Set([
:SettlementCount,
:CityCount,
:RoadCount,
:MaxRoadLength,
:SumWoodDiceWeight,
:SumBrickDiceWeight,
:SumPastureDiceWeight,
:SumStoneDiceWeight,
:SumGrainDiceWeight,
:CountPortWood,
:CountPortBrick,
:CountPortPasture,
:CountPortStone,
:CountPortGrain,
:CountHandWood,
:CountHandBrick,
:CountHandPasture,
:CountHandStone,
:CountHandGrain,
:CountTotalWood,
:CountTotalBrick,
:CountTotalPasture,
:CountTotalStone,
:CountTotalGrain,
:CountKnight,
:CountMonopoly,
:CountYearOfPlenty,
:CountRoadBuilding,
#:HasLargestArmy,
#:HasLongestRoad,
:CountVictoryPoint
])

"""
    `test_feature_perturbations()`

This tests that the value estimations behave as expected.  Each feature in `features_increasing_good` should be positively correlated with value, since they are strictly good for the player when manipulated independently.

So this test applies one-at-a-time perturbations to the features, checking that the value changes in the correct direction.
"""
function test_feature_perturbations(features, features_increasing_good, configs, max_perturbation = 3)
    state_to_value = read_values_file(configs["PlayerSettings"]["STATE_VALUES"], 100)
    
    feature_vec = generate_realistic_features(features)
    feature_values = [f[2] for f in feature_vec]

    value_player = TemporalDifferencePlayer(MaxValueMarkovPolicy, :Blue, state_to_value, Dict{UInt64, Float64}(), configs)
    reward_player = TemporalDifferencePlayer(MaxRewardMarkovPolicy, :Red, state_to_value, Dict{UInt64, Float64}(), configs)
    
    current_state = MarkovState(reward_player.process, feature_vec, value_player.model)
    value = get_state_optimizing_quantity(value_player.process, value_player.policy, current_state)
    reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, current_state)
    model_proba = predict_model(value_player.model, feature_vec)
    
    fails_v = Dict()
    fails_r = Dict()
    fails_m = Dict()
    #fails_2 = []
    #fails_1 = []

    for epsilon in max_perturbation:-1:1
        fails_v[epsilon] = []
        fails_r[epsilon] = []
        fails_m[epsilon] = []
        for (i,name) in enumerate(features)
            if ~(name in features_increasing_good)
                println("ignoring $name")
                continue
            end
            
            feature_values[i] = epsilon
            feature_vec = [Pair(f,v) for (f,v) in zip(features, feature_values)]
            next_state = MarkovState(reward_player.process, feature_vec, value_player.model)
            next_value = get_state_optimizing_quantity(value_player.process, value_player.policy, next_state)
            next_reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, next_state)
            next_model_proba = predict_model(value_player.model, feature_vec)
            
            @test next_state.reward == next_reward
            
            #println("evaluating $name + $epsilon")
            if next_value < value
                push!(fails_v[epsilon], name)
            end
            if next_model_proba < model_proba
                push!(fails_m[epsilon], name)
            end
            if next_reward < reward
                push!(fails_r[epsilon], name)
            end
            """
            @test next_model_proba >= model_proba
            @test next_reward >= reward
            @test next_value >= value
            """
            
            feature_values[i] = 0.0
        end
    end
    return fails_m, fails_r, fails_v
end

function generate_realistic_features(features)
    feature_values = Dict([f => 0.0 for f in features])

    for f in features
        if haskey(feature_library, f)
            mn = feature_library[f].min
            mx = feature_library[f].max

            feature_values[f] = sample(range(mn,mx), 1, replace=false)[1]
        else
            feature_values[f] = sample(range(0,10), 1, replace=false)[1]
        end
    end
    return [Pair(f,v) for (f,v) in collect(feature_values)]
end
end

@testitem "jet_fails" tags=[:slow] setup=[global_test_setup] begin
    rep = report_package(CatanLearning;
    ignored_modules=())

    #@show length(JET.get_reports(rep))
    #@show rep
    reports = JET.get_reports(rep)
    max_num = 66
    println("length(JET.get_reports(rep)) = $(length(reports)) / $max_num")
    @test length(reports) <= max_num
end

function test_evolving_robot_game(neverend, configs)
    team_and_playertype = [
                          (:blue, EmpathRobotPlayer),
                          (:cyan, EmpathRobotPlayer),
                          (:green, EmpathRobotPlayer),
                          (:red, EmpathRobotPlayer)
            ]
    players = setup_players(team_and_playertype, configs)
    test_automated_game(neverend, players, configs)
end

function empath_player(configs)
    player = EmpathRobotPlayer(:red)
    board = Board(configs)
    p = predict_model(player.model, board, player)
    return player, board, p
end


@testitem "model_caching" setup=[global_test_setup] begin
    @test CatanLearning.get_ml_cache_config(configs, :blue, "TEST_KEY") === nothing
    CatanLearning.update_ml_cache!(configs, :blue, "TEST_KEY", 50)
    CatanLearning.get_ml_cache_config(configs, :blue, "TEST_KEY") == 50
    CatanLearning.get_ml_cache_config(configs, :green, "TEST_KEY") === nothing
end

@testitem "learning_player_base_actions" setup=[global_test_setup] begin
    player = EmpathRobotPlayer(:Red, configs)
    player2 = DefaultRobotPlayer(:Blue, configs)
    players = [player, player2]
    game = Game(players, configs)
    board = Board(configs)
    # get_legal_action_sets(board::Board, players::AbstractVector{PlayerPublicView}, player::Player, pre_actions::Set{PreAction})::Vector{AbstractActionSet}

    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Wood)
    PlayerApi.give_resource!(player.player, :Brick)
    PlayerApi.give_resource!(player.player, :Pasture)
    candidates = BoardApi.get_admissible_settlement_locations(board, player.player.team, true)
    pre_actions = Set([PreAction(:ConstructSettlement, candidates)])
    legal_actions = get_legal_action_sets(board, PlayerPublicView.(game.players), player.player, pre_actions)
    
    Catan.ACTIONS_DICTIONARY[:ConstructSettlement](game, board, player, legal_actions[1].actions[1].args, false)
    BoardApi.print_board(board)
    road_candidates = BoardApi.get_admissible_road_locations(board, player.player.team, true)
    @test ~isempty(road_candidates)

    PlayerApi.give_resource!(player.player, :Wood)
    PlayerApi.give_resource!(player.player, :Brick)
    Catan.choose_road_location(board, PlayerPublicView.(game.players), player, road_candidates)


    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Stone)
    PlayerApi.give_resource!(player.player, :Stone)
    PlayerApi.give_resource!(player.player, :Stone)
    candidates = BoardApi.get_admissible_city_locations(board, player.player.team)
    pre_actions = Set([PreAction(:ConstructCity, candidates)])
    legal_actions = get_legal_action_sets(board, PlayerPublicView.(game.players), player.player, pre_actions)
    Catan.ACTIONS_DICTIONARY[:ConstructCity](game, board, player, legal_actions[1].actions[1].args)

    return legal_actions
end



@testitem "compute_features" setup=[global_test_setup] begin
    #players = setup_players()
    player = Catan.DefaultRobotPlayer(:Blue, configs)
    board = Board(configs)
    compute_features(board, player.player)
end

@testitem "feature_perturbations" setup=[global_test_setup] begin
    (fails_m, fails_r, fails_v) = test_feature_perturbations(features, features_increasing_good, configs)
    println("model fails with +3 perturbation $(length(fails_m[3])): $(fails_m[3])")
    println("model fails with +2 perturbation $(length(fails_m[2])): $(fails_m[2])")
    println("model fails with +1 perturbation $(length(fails_m[1])): $(fails_m[1])")
    
    #=
    println("reward fails with +3 perturbation: $(fails_r[3])")
    println("reward fails with +2 perturbation: $(fails_r[2])")
    println("reward fails with +1 perturbation: $(fails_r[1])")
    println("value fails with +3 perturbation: $(fails_v[3])")
    println("value fails with +2 perturbation: $(fails_v[2])")
    println("value fails with +1 perturbation: $(fails_v[1])")
    =#
end


@testitem "choose_road_location" setup=[global_test_setup] begin
    player1 = DefaultRobotPlayer(:Test1, configs)
    player2 = DefaultRobotPlayer(:Test2, configs)
    players = Vector{PlayerType}([player1, player2])
    board = Board(configs)
    player1 = DefaultRobotPlayer(:Test1, configs)
    #choose_road_location(board::Board, PlayerPublicView.(players), player1, })
end

@testitem "stackoverflow_knight" setup=[global_test_setup] begin
    copied_configs = deepcopy(configs)
    copied_configs["PlayerSettings"]["SEARCH_DEPTH"] = 2
    player = EmpathRobotPlayer(:Blue, configs)
    players = Vector{PlayerType}([
                                  player, 
                                  DefaultRobotPlayer(:Test2, copied_configs),
                                  DefaultRobotPlayer(:Test3, copied_configs),
                                  DefaultRobotPlayer(:Test4, copied_configs),
                                 ])
    board = Board(copied_configs)
    game = Game(players, copied_configs)
    PlayerApi.add_devcard!(player.player, :Knight)
    actions = get_legal_actions(game, board, player.player)
    println(actions)
    choice = Catan.choose_next_action(board, PlayerPublicView.(players), player, actions)
end

@testitem "empath_road_building" setup=[global_test_setup] begin
    players = Vector{PlayerType}([
                                  EmpathRobotPlayer(:Blue, configs), 
                                  DefaultRobotPlayer(:Test2, configs),
                                  DefaultRobotPlayer(:Test3, configs),
                                  DefaultRobotPlayer(:Test4, configs),
                                 ])
    player = players[1]
    board = Board(configs)
    
    #BoardApi.build_settlement!(board, :Blue, (1,1))
    Catan.do_road_building_action(board, PlayerPublicView.(players), player)
end

@testitem "action_interface" setup=[global_test_setup] begin
    players = Vector{PlayerType}([
                                  EmpathRobotPlayer(:Blue, configs), 
                                  DefaultRobotPlayer(:Test2, configs),
                                  DefaultRobotPlayer(:Test3, configs),
                                  DefaultRobotPlayer(:Test4, configs),
                                 ])
    player = players[1]
    board = Board(configs)

    BoardApi.build_settlement!(board, :Blue, (1,1))
    #admissible_roads = BoardApi.get_admissible_road_locations(board, player.player.team, true)
    admissible_roads = Vector{Tuple}([((1,1), (1,2), false), ((1,1),(2,2), false)])
    game = Game(players, configs)
    #actions = Set([PreAction(:BuyDevCard), PreAction(:ConstructRoad, admissible_roads)])

    # Build road
    PlayerApi.give_resource!(player.player, :Brick)
    PlayerApi.give_resource!(player.player, :Wood)
    
    # Dev card
    PlayerApi.give_resource!(player.player, :Pasture)
    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Stone)

    actions = get_legal_actions(game, board, player.player)::Set{PreAction}

    # collected to compare the custom == def for PreAction, not the hashed value
    collected_actions = collect(actions)

    @test PreAction(:BuyDevCard) in collected_actions
    @test PreAction(:ConstructRoad, admissible_roads) in collected_actions
    @test PreAction(:DoNothing) in collected_actions
    @test PreAction(:ProposeTrade) in collected_actions

    action_sets = get_legal_action_sets(board, PlayerPublicView.(players), player.player, actions)
    #best_action = get_best_action(board, players, player, actions)
    
#=
    for set in action_sets
        @test action in action_sets
    end
    =#
end

@testitem "perturbations" setup=[global_test_setup] begin
    feats = [1.0, 2.0, 3.0]
    model = CatanLearning.LinearModel(feats)

    CatanLearning.add_perturbation!(model, 0.1)
    @test model.weights[1] != feats[1]
    @test model.weights[2] != feats[2]
    @test model.weights[3] != feats[3]
end

@testitem "player_implementation_default" setup=[global_test_setup] begin
    test_player_implementation(Catan.DefaultRobotPlayer, configs)
end
@testitem "player_implementation_empath" setup=[global_test_setup] begin
    test_player_implementation(EmpathRobotPlayer, configs)
end
@testitem "player_implementation_hybrid" setup=[global_test_setup] begin
    test_player_implementation(HybridPlayer, configs)
end
@testitem "player_implementation_td" setup=[global_test_setup] begin
    test_player_implementation(TemporalDifferencePlayer, configs)
end

function run_tests()
    @run_package_tests filter=ti->doset(ti)
    configs = parse_configs("Configuration.toml")
    rm("savefile.txt")
end

run_tests()
