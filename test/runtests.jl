using Test
using StatsBase
using Logging

using CatanLearning:
    compute_features
#include("../src/CatanLearning.jl")


function test_evolving_robot_game(neverend)
    team_and_playertype = [
                          (:blue, EmpathRobotPlayer),
                          (:cyan, MutatedEmpathRobotPlayer),
                          (:green, EmpathRobotPlayer),
                          (:red, MutatedEmpathRobotPlayer)
            ]
    players = Catan.setup_players(team_and_playertype)
    Catan.test_automated_game(neverend, players)
end

function empath_player()
    player = EmpathRobotPlayer(:red)
    board = read_map(SAMPLE_MAP)
    p = predict_model(player.machine, board, player)
    return player, board, p
end


features = [
:SettlementCount,
:CityCount,
:RoadCount,
:MaxRoadLength,
:SumWoodDiceWeight,
:SumBrickDiceWeight,
:SumPastureDiceWeight,
:SumStoneDiceWeight,
:SumGrainDiceWeight,
:PortWood,
:PortBrick,
:PortPasture,
:PortStone,
:PortGrain,
:CountWood,
:CountBrick,
:CountPasture,
:CountStone,
:CountGrain,
:CountKnight,
:CountMonopoly,
:CountYearOfPlenty,
:CountRoadBuilding,
:CountVictoryPoint
]
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
:PortWood,
:PortBrick,
:PortPasture,
:PortStone,
:PortGrain,
:CountWood,
:CountBrick,
:CountPasture,
:CountStone,
:CountGrain,
:CountKnight,
:CountMonopoly,
:CountYearOfPlenty,
:CountRoadBuilding,
:CountVictoryPoint
])

MAIN_DATA_DIR = "../data/"

function test_compute_features()
    #players = setup_players()
    player = DefaultRobotPlayer(:Blue)
    board = read_map("../../CatanEngine.jl/data/sample.csv")
    compute_features(board, player)
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
function test_feature_perturbations(features, features_increasing_good, max_perturbation = 3)
    state_to_value = read_values_file("$MAIN_DATA_DIR/state_values.csv")
    
    feature_vec = generate_realistic_features(features)
    feature_values = [f[2] for f in feature_vec]

    value_player = TemporalDifferencePlayer(MaxValueMarkovPolicy, :Blue, state_to_value, Dict{UInt64, Float64}())
    reward_player = TemporalDifferencePlayer(MaxRewardMarkovPolicy, :Red, state_to_value, Dict{UInt64, Float64}())
    
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


function run_tests(neverend = false)
    test_compute_features()
    """
    (fails_m, fails_r, fails_v) = test_feature_perturbations(features, features_increasing_good)
    println("model fails with +3 perturbation $(length(fails_m[3])): $(fails_m[3])")
    println("model fails with +2 perturbation $(length(fails_m[2])): $(fails_m[2])")
    println("model fails with +1 perturbation $(length(fails_m[1])): $(fails_m[1])")
    println("reward fails with +3 perturbation: $(fails_r[3])")
    println("reward fails with +2 perturbation: $(fails_r[2])")
    println("reward fails with +1 perturbation: $(fails_r[1])")
    println("value fails with +3 perturbation: $(fails_v[3])")
    println("value fails with +2 perturbation: $(fails_v[2])")
    println("value fails with +1 perturbation: $(fails_v[1])")
    test_evolving_robot_game(neverend)
    """
end

