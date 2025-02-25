using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: deepcopy,hash
using DelimitedFiles

abstract type LearningPlayer <: RobotPlayer
end

mutable struct EmpathRobotPlayer <: LearningPlayer 
    player::Player
    machine::Machine
end

mutable struct MutatedEmpathRobotPlayer <: LearningPlayer 
    player::Player
    machine::Machine
    mutation::Dict #{Symbol, AbstractFloat}
end

mutable struct TemporalDifferencePlayer <: LearningPlayer
    player::Player
    machine::Machine
    process::MarkovRewardProcess
    policy::MarkovPolicy
    io_config::IoConfig
end

function TemporalDifferencePlayer(team::Symbol)
    io_config = IoConfig()
    state_to_value = read_values_file(io_config.values)
    return TemporalDifferencePlayer(team, state_to_value)
end
function TemporalDifferencePlayer(team::Symbol, master_state_to_value::Dict{UInt64, Float64}, new_state_to_value::Dict{UInt64, Float64})
    Tree = load_tree_model()
    io_config = IoConfig()
    tree = Base.invokelatest(Tree,
        max_depth = 6,
        min_gain = 0.0,
        min_records = 2,
        max_features = 0,
        splitting_criterion = BetaML.Utils.gini)
    machine = try_load_model_from_csv(tree, io_config.model, io_config.features)

    process = MarkovRewardProcess(0.5, 0.1, 0.5, 0.5, master_state_to_value, new_state_to_value)
    policy = MarkovPolicy(machine)
    TemporalDifferencePlayer(Player(team), machine, process, policy, io_config)
end

MutatedEmpathRobotPlayer(team::Symbol) = MutatedEmpathRobotPlayer(team, "../../features.csv", Dict{Symbol, AbstractFloat}())
MutatedEmpathRobotPlayer(team::Symbol, mutation::Dict) = MutatedEmpathRobotPlayer(team, "../../features.csv", mutation)

function MutatedEmpathRobotPlayer(team::Symbol, features_file_name::String, mutation::Dict)
    Tree = load_tree_model()
    tree = Base.invokelatest(Tree,
        max_depth = 6,
        min_gain = 0.0,
        min_records = 2,
        max_features = 0,
        splitting_criterion = BetaML.Utils.gini)
    MutatedEmpathRobotPlayer(Player(team), try_load_model_from_csv(tree, "$(DATA_DIR)/model.jls", "$(DATA_DIR)/features.csv"), mutation)
end

function Base.deepcopy(player::MutatedEmpathRobotPlayer)
    return MutatedEmpathRobotPlayer(deepcopy(player.player), deepcopy(player.machine), deepcopy(player.mutation)) #TODO needto deepcopy the machine?
end

function Base.deepcopy(player::TemporalDifferencePlayer)
    # Note, we deepcopy only the player data, while the RL data should persist in order to pass updates the state info properly
    return TemporalDifferencePlayer(deepcopy(player.player), player.machine, player.process, player.policy, player.io_config)
end

EmpathRobotPlayer(team::Symbol) = EmpathRobotPlayer(team, "../../features.csv")
function EmpathRobotPlayer(team::Symbol, features_file_name::String)
    Tree = load_tree_model()
    tree = Base.invokelatest(Tree,
        max_depth = 6,
        min_gain = 0.0,
        min_records = 2,
        max_features = 0,
        splitting_criterion = BetaML.Utils.gini)
    EmpathRobotPlayer(Player(team), try_load_model_from_csv(tree, "$(DATA_DIR)/model.jls", "$(DATA_DIR)/features.csv"))
end


function Base.deepcopy(player::EmpathRobotPlayer)
    return EmpathRobotPlayer(deepcopy(player.player), player.machine) #TODO needto deepcopy the machine?
end

inner_player(player::EmpathRobotPlayer)::Player = p -> p.player
inner_player(player::MutatedEmpathRobotPlayer)::Player = p -> p.player
inner_player(player::TemporalDifferencePlayer)::Player = p -> p.player
ml_machine(player::LearningPlayer)::Machine = p -> p.machine
ml_machine(player::EmpathRobotPlayer)::Machine = p -> p.machine
ml_machine(player::MutatedEmpathRobotPlayer)::Machine = p -> p.machine
ml_machine(player::TemporalDifferencePlayer)::Machine = p -> p.machine
