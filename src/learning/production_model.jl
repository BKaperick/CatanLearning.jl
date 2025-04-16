using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
using MLJDecisionTreeInterface
using DecisionTree

function create_new_model(features_file, model_path)
    serialize_model_from_csv_features(load_tree_model(), features_file, model_path)
end

function load_tree_model()
    return (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
end

function try_load_model_from_csv(player_configs::Dict)::Machine
    try_load_model_from_csv(player_configs["MODEL"],  player_configs["FEATURES"])
end

function try_load_public_model_from_csv(player_configs::Dict)::Machine
    try_load_model_from_csv(player_configs["PUBLIC_MODEL"],  player_configs["PUBLIC_FEATURES"])
end

"""
    try_load_model_from_csv(tree, model_file_name, features_file_name)

If the serialized file exists, then load it.  If not, train a new model and 
serialize it before returning it to caller.
"""
function try_load_model_from_csv(model_file_name, features_file_name)::Machine
    @info "Looking for model stored in $model_file_name"
    if isfile(model_file_name)
        @info "Found model stored in $model_file_name"
        return load_model_from_csv(model_file_name)
    end
    @info "Not found, let's try to train a new model from features in $features_file_name"
    return serialize_model_from_csv_features(load_tree_model(), features_file_name, model_file_name)
end

function load_model_from_csv(model_file_name)::Machine
    return machine(model_file_name)
end

train_model_from_csv(tree, player_configs::Dict) = train_model_from_csv(tree, player_configs["FEATURES"])
train_public_model_from_csv(tree, player_configs::Dict) = train_model_from_csv(tree, player_configs["PUBLIC_FEATURES"])

function coerce_feature_types!(df)
    for (name,feat) in CatanLearning.feature_library
        if string(name) in names(df)
            df[!,name] = convert(Vector{feat.type}, df[!,name])
        end
    end
    coerce!(df, :WonGame => OrderedFactor{2})
end

function load_typed_features_from_csv(features_csv)
    data, header = readdlm(features_csv, ',', header=true)
    df = DataFrame(data, vec(header))
    coerce_feature_types!(df)

    select!(df, Not([
        :CountHandWood,
        :CountHandBrick,
        :CountHandPasture,
        :CountHandStone,
        :CountHandGrain,
        :HasMostPoints,
        :CountVictoryPoint
        ]))

    df = DFM.@transform(df, :WonGame)
    return df
end

function train_model_from_csv(tree, features_csv; num_tuning_iterations = 100)
    df = load_typed_features_from_csv(features_csv)
    df_train, df_test = partition(df, 0.7, rng=123)

    y, X = unpack(df_train, ==(:WonGame));
    y_test, X_test = unpack(df_test, ==(:WonGame));

    thresholded_tree = tree
    #thresholded_tree = MLJ.BinaryThresholdPredictor(model=tree, threshold=0.5)
    ranges = [
        #range(thresholded_tree, :threshold, lower=0.1, upper=0.9),
        range(thresholded_tree, :(min_purity_increase), lower=0.0, upper=0.9),
        range(thresholded_tree, :(min_samples_leaf), lower=4, upper=10),
        range(thresholded_tree, :(min_samples_split), lower=2, upper=8),
        #range(thresholded_tree, :(partial_sampling), lower=0.5, upper=0.9),
        range(thresholded_tree, :(n_trees), lower=5, upper=20)
    ]

    tuned_tree = TunedModel(
        thresholded_tree,
        tuning=RandomSearch(),
        resampling=CV(nfolds=6),
        range = ranges,
        measure = MatthewsCorrelation(),
        n=num_tuning_iterations
    )
    
    tuned_mach = machine(tuned_tree, X, y) |> MLJ.fit!

    append!(X, X_test)
    append!(y, y_test)
    final_mach = machine(report(tuned_mach).best_model, X, y) |> MLJ.fit!

    return final_mach 
end

function serialize_model_from_csv_features(tree, csv_name, model_path)
    mach = train_model_from_csv(tree, csv_name)
    MLJ.save(model_path, mach)
    return mach
end

"""
    `train_and_serialize_model(features_csv, output_path)`

This is the access point for re-training a model based on new features or engine bug fixes.
"""
function train_and_serialize_model(features_csv, output_path)
    Tree = @load RandomForestClassifier pkg=DecisionTree verbosity=0
    tree = Base.invokelatest(Tree)
    tuned_mach = train_model_from_csv(tree, features_csv)
    MLJ.save(output_path, tuned_mach)
end
