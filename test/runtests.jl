using Test
using StatsBase
using Logging
using Catan
using Catan: Game, Board, Player, PlayerType, PlayerApi, BoardApi, GameApi,
             DefaultRobotPlayer, PlayerPublicView, 
             ALL_ACTIONS, choose_accept_trade, get_legal_actions,
read_map,
load_gamestate!,
reset_savefile!,
test_player_implementation,
setup_players,
setup_and_do_robot_game,
test_automated_game,
#configs,
#player_configs,
#logger,
parse_configs

using CatanLearning:
    compute_features,
    MutatedEmpathRobotPlayer,EmpathRobotPlayer,

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

function test_jet_fails()
    rep = report_package(CatanLearning;
    ignored_modules=())
    @test length(JET.get_reports(rep)) <= 31
end

function test_evolving_robot_game(neverend, configs)
    team_and_playertype = [
                          (:blue, EmpathRobotPlayer),
                          (:cyan, MutatedEmpathRobotPlayer),
                          (:green, EmpathRobotPlayer),
                          (:red, MutatedEmpathRobotPlayer)
            ]
    players = setup_players(team_and_playertype, configs)
    test_automated_game(neverend, players, configs)
end

function empath_player(configs)
    player = EmpathRobotPlayer(:red)
    board = read_map(configs)
    p = predict_model(player.machine, board, player)
    return player, board, p
end


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

function test_compute_features(configs)
    #players = setup_players()
    player = Catan.DefaultRobotPlayer(:Blue, configs)
    board = Catan.read_map(configs)
    compute_features(board, player.player)
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
    
    current_state = MarkovState(feature_vec)
    value = get_state_optimizing_quantity(value_player.process, value_player.policy, current_state)
    reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, current_state)
    model_proba = predict_model(value_player.machine, feature_vec)
    
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
            next_state = MarkovState(feature_vec)
            feature_values[i] = epsilon
            feature_vec = [Pair(f,v) for (f,v) in zip(features, feature_values)]
            next_state = MarkovState(feature_vec)
            next_value = get_state_optimizing_quantity(value_player.process, value_player.policy, next_state)
            next_reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, next_state)
            next_model_proba = predict_model(value_player.machine, feature_vec)
            
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

function test_choose_road_location(configs)

    player1 = DefaultRobotPlayer(:Test1)
    player2 = DefaultRobotPlayer(:Test2)
    players = Vector{PlayerType}([player1, player2])
    board = read_map(configs)
    player1 = DefaultRobotPlayer(:Test1)
choose_road_location(board::Board, players::Vector{PlayerPublicView}, player::LearningPlayer, candidates::Vector{Vector{Tuple{Int, Int}}})
end

function test_stackoverflow_knight(main_configs)
    configs = deepcopy(main_configs)
    configs["PlayerSettings"]["SEARCH_DEPTH"] = 2
    player = EmpathRobotPlayer(:Blue, configs)
    players = Vector{PlayerType}([
                                  player, 
                                  DefaultRobotPlayer(:Test2, configs),
                                  DefaultRobotPlayer(:Test3, configs),
                                  DefaultRobotPlayer(:Test4, configs),
                                 ])
    board = read_map(configs)
    game = Game(players, configs)
    PlayerApi.add_devcard!(player.player, :Knight)
    actions = get_legal_actions(game, board, player.player)
    choice = Catan.choose_next_action(board, PlayerPublicView.(players), player, actions)
end

function test_empath_road_building(configs)
    players = Vector{PlayerType}([
                                  EmpathRobotPlayer(:Blue, configs), 
                                  DefaultRobotPlayer(:Test2, configs),
                                  DefaultRobotPlayer(:Test3, configs),
                                  DefaultRobotPlayer(:Test4, configs),
                                 ])
    player = players[1]
    board = read_map(configs)
    
    #BoardApi.build_settlement!(board, :Blue, (1,1))
    Catan.do_road_building_action(board, PlayerPublicView.(players), player)
end

function test_action_interface(configs)
    players = Vector{PlayerType}([
                                  EmpathRobotPlayer(:Blue, configs), 
                                  DefaultRobotPlayer(:Test2, configs),
                                  DefaultRobotPlayer(:Test3, configs),
                                  DefaultRobotPlayer(:Test4, configs),
                                 ])
    player = players[1]
    board = read_map(configs)

    admissible_roads = BoardApi.get_admissible_road_locations(board, player.player.team)
    actions = Set([PreAction(:BuyDevCard), PreAction(:ConstructRoad, admissible_roads)])
    BoardApi.build_settlement!(board, :Blue, (1,1))

    # Build road
    PlayerApi.give_resource!(player.player, :Brick)
    PlayerApi.give_resource!(player.player, :Wood)
    
    # Dev card
    PlayerApi.give_resource!(player.player, :Pasture)
    PlayerApi.give_resource!(player.player, :Grain)
    PlayerApi.give_resource!(player.player, :Stone)
    action_sets = get_legal_action_sets(board, PlayerPublicView.(players), player.player, actions)
    #best_action = get_best_action(board, players, player, actions)
end

function run_tests(neverend = false)
    configs = parse_configs("Configuration.toml")
    test_jet_fails()
    test_stackoverflow_knight(configs)
    test_empath_road_building(configs)
    test_action_interface(configs)
    test_player_implementation(Catan.DefaultRobotPlayer, configs)
    test_player_implementation(EmpathRobotPlayer, configs)
    test_player_implementation(MutatedEmpathRobotPlayer, configs)
    test_player_implementation(TemporalDifferencePlayer, configs)
    test_compute_features(configs)
    test_evolving_robot_game(neverend, configs)
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

if length(ARGS) > 1
    run_tests(ARGS[1])
else
    run_tests()
end
