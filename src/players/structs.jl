using MLJ
using DataFrames
import DataFramesMeta as DFM
import Base: deepcopy
using DelimitedFiles

mutable struct MutatedEmpathRobotPlayer <: RobotPlayer
    player::Player
    machine::Machine
    mutation::Dict #{Symbol, AbstractFloat}
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

mutable struct EmpathRobotPlayer <: RobotPlayer
    player::Player
    machine::Machine
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
