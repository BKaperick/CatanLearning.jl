using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
using MLJDecisionTreeInterface
using DecisionTree
using LinearAlgebra

function load_tree_model()
    return (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
end

function try_load_model_from_csv(team::Symbol, configs::Dict)::Machine
    try_load_serialized_model_from_csv(get_player_config(configs, "MODEL", team),  get_player_config(configs, "FEATURES", team))
end

function try_load_linear_model_from_csv(team::Symbol, configs::Dict)::Vector{Float64}
    model_path = get_player_config(configs, "MODEL", team)
    @info "Looking for linear model stored in $model_path"
    if isfile(model_path)
        @info "Found model stored in $model_path"
        weights = CSV.read(model_path, DataFrame)
        return weights[!, :Weights]
    end
    @assert false "Not found"
end

function try_load_public_model_from_csv(team::Symbol, configs::Dict)::Machine
    try_load_serialized_model_from_csv(get_player_config(configs, "PUBLIC_MODEL", team),  get_player_config(configs, "PUBLIC_FEATURES", team))
end

"""
    try_load_model_from_csv(tree, model_file_name, features_file_name)

If the serialized file exists, then load it.  If not, train a new model and 
serialize it before returning it to caller.
"""
function try_load_serialized_model_from_csv(model_file_name::String, features_file_name::String)::Machine
    @info "Looking for model stored in $model_file_name"
    if isfile(model_file_name)
        @info "Found model stored in $model_file_name"
        return load_model_from_csv(model_file_name)
    end
    @info "Not found, let's try to train a new model from features in $features_file_name"
    train_and_serialize_model(features_file_name, model_file_name; num_tuning_iterations = 100)
end

function load_model_from_csv(model_file_name)::Machine
    return machine(model_file_name)
end

function coerce_feature_types!(df)
    for (name,feat) in CatanLearning.feature_library
        if string(name) in names(df)
            try
                df[!,name] = convert(Vector{feat.type}, df[!,name])
            catch e
                @warn "Failed to convert $name to $(feat.type): \n$e"
            end
        end
    end
    coerce!(df, :WonGame => OrderedFactor{2})
end
function filter_bad_features!(df)
    features_to_exclude = [
        :CountHandWood,
        :CountHandBrick,
        :CountHandPasture,
        :CountHandStone,
        :CountHandGrain,
        :HasMostPoints,
        :NumberOfTurns,
        :CountVictoryPoint
        ]
    for feat in features_to_exclude
        if String(feat) in names(df)
            #@debug "removing $feat from features while loading"
            select!(df, Not([feat]))
        end
    end
end

function load_typed_features_from_csv(features_csv)
    data, header = readdlm(features_csv, ',', header=true)
    df = DataFrame(data, vec(header))
    coerce_feature_types!(df)
    filter_bad_features!(df)

    df = DFM.@transform(df, :WonGame)
    return df
end

function train_model_from_csv(tree, features_csv; num_tuning_iterations = 100)
    df = load_typed_features_from_csv(features_csv)
    df_train, df_test = partition(df, 0.7, rng=123)

    y, X = unpack(df_train, ==(:WonGame));
    y_test, X_test = unpack(df_test, ==(:WonGame));
    
    @info "Training model on $(length(names(X))) features in $features_csv"
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

"""
    `train_and_serialize_model(features_csv, output_path)`

This is the access point for re-training a model based on new features or engine bug fixes.
"""
function train_and_serialize_model(features_csv::String, output_path::String; num_tuning_iterations = 100)
    tree = load_tree_model()
    tuned_mach = train_model_from_csv(tree, features_csv, num_tuning_iterations = num_tuning_iterations)
    @info "Serializing model trained on $features_csv into $output_path"
    MLJ.save(output_path, tuned_mach)
end

function train_and_serialize_linear_model(features_csv::String, output_path::String)
    tuned_mach = train_linear_model_from_csv(features_csv)
    @info "Serializing linear model trained on $features_csv into $output_path"
    
end

function train_linear_model_from_csv(features_csv)
    df = load_typed_features_from_csv(features_csv)

    M = df |> Tables.matrix
    U,S,V = svd(M)



    y, X = unpack(df_train, ==(:WonGame));
    y_test, X_test = unpack(df_test, ==(:WonGame));
    
    @info "Training model on $(length(names(X))) features in $features_csv"
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