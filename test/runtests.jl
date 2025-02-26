using Test

include("../src/CatanLearning.jl")


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

"""
    `test_feature_perturbations()`

This tests that the value estimations behave as expected.  Each feature in `features_increasing_good` should be positively correlated with value, since they are strictly good for the player when manipulated independently.

So this test applies one-at-a-time perturbations to the features, checking that the value changes in the correct direction.
"""
function test_feature_perturbations(features, features_increasing_good, max_perturbation = 3)
    state_to_value = read_values_file("$MAIN_DATA_DIR/state_values.csv")
    
    feature_vec = [Pair(f,0.0) for f in features]
    feature_values = [0.0 for f in features]

    value_player = TemporalDifferencePlayer(MaxValueMarkovPolicy, :Blue, state_to_value, Dict{UInt64, Float64}())
    reward_player = TemporalDifferencePlayer(MaxRewardMarkovPolicy, :Red, state_to_value, Dict{UInt64, Float64}())
    
    current_state = MarkovState(feature_vec)
    value = get_state_optimizing_quantity(value_player.process, value_player.policy, current_state)
    reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, current_state)
    model_proba = predict_model(value_player.machine, feature_vec)
    
    fails = Dict()
    #fails_2 = []
    #fails_1 = []

    for epsilon in max_perturbation:-1:1
        fails[epsilon] = []
        for (i,name) in enumerate(features)
            if ~(name in features_increasing_good)
                println("ignoring $name")
                continue
            end
            
            feature_values[i] = epsilon
            feature_vec = [Pair(f,v) for (f,v) in zip(features, feature_values)]
            next_state = MarkovState(feature_vec)
            next_value = get_state_optimizing_quantity(value_player.process, value_player.policy, next_state)
            next_reward = get_state_optimizing_quantity(reward_player.process, reward_player.policy, next_state)
            next_model_proba = predict_model(value_player.machine, feature_vec)
            
            #println("evaluating $name + $epsilon")
            if next_value < value
                push!(fails[epsilon], name)
            end
            """
            if next_model_proba < model_proba
                push!(fails[epsilon], name)
            end
            if next_reward < reward
                push!(fails[epsilon], name)
            end
            @test next_model_proba >= model_proba
            @test next_reward >= reward
            @test next_value >= value
            """
            
            feature_values[i] = 0.0
        end
    end
    return fails
end


fails = test_feature_perturbations(features, features_increasing_good)
#test_evolving_robot_game(neverend)
println("Fails with +3 perturbation: $(fails[3])")
println("Fails with +2 perturbation: $(fails[2])")
println("Fails with +1 perturbation: $(fails[1])")
