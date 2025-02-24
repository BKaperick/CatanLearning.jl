using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: deepcopy
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
