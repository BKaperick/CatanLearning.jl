using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
using MLJDecisionTreeInterface
using DecisionTree
using LinearAlgebra
using Distributions

function load_tree_model()
    return (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
end

function get_ml_cache_config(configs::Dict, team::Symbol, key::String)
    if !haskey(configs, "ML_CACHE")
        configs["ML_CACHE"] = Dict(["PlayerSettings"=>Dict()])
    end
    if haskey(configs["ML_CACHE"]["PlayerSettings"], String(team)) && 
        haskey(configs["ML_CACHE"]["PlayerSettings"][String(team)], key)
        return configs["ML_CACHE"]["PlayerSettings"][String(team)][key]
    end
    return
end

function update_ml_cache!(configs::Dict, team, key::String, obj)
    if !haskey(configs, "ML_CACHE")
        configs["ML_CACHE"] = Dict(["PlayerSettings"=>Dict()])
    end
    if !haskey(configs["ML_CACHE"]["PlayerSettings"], String(team))
        configs["ML_CACHE"]["PlayerSettings"][String(team)] = Dict()
    end
    configs["ML_CACHE"]["PlayerSettings"][String(team)][key] = obj
end

function try_load_serialized_model(team::Symbol, configs::Dict, model_key="MODEL", features_key = "FEATURES")::DecisionModel
    cached = get_ml_cache_config(configs, team, model_key)::Union{DecisionModel, Nothing}
    if cached !== nothing
        return cached
    end
    model_path = get_player_config(configs, model_key, team)
    features_path = get_player_config(configs, features_key, team)
    model = load_or_train_serialized_model(model_path, features_path, get_player_config(configs, "$(model_key)_FORCE_RETRAIN", team))
    update_ml_cache!(configs, team, model_key, model)
    return model
end

function try_load_serialized_public_model(team::Symbol, configs::Dict)::DecisionModel
    try_load_serialized_model(team, configs, "PUBLIC_MODEL", "PUBLIC_FEATURES")
end

"""
    load_or_train_serialized_model(model_path::String, features_path::String, force_retrain::Bool)::DecisionModel

If the serialized file exists, then load it.  If not, train a new model and 
serialize it before returning it to caller.  Use `force_retrain` to force training and serialization of a new model.
If `model_path` ends in csv or jls we have dedicated training and serialization implementation.
"""
function load_or_train_serialized_model(model_path::String, features_path::String, force_retrain::Bool)::DecisionModel
    if model_path == ""
        return EmptyModel()
    elseif endswith(lowercase(model_path), "csv")
        return load_or_train_serialized_model_from_csv(model_path, features_path, force_retrain)
    elseif endswith(lowercase(model_path), "jls")
        return load_or_train_serialized_model_from_jls(model_path, features_path, force_retrain)
    else
        throw(ArgumentError("Unrecognized model serialization formation for model $model_path (Only .csv and .jls file deserialization is currentlly implemented)"))
    end
end

function load_or_train_serialized_model_from_csv(model_path::String, features_path::String, force_retrain::Bool)
    if isfile(model_path) && !force_retrain
        @info "Found CSV model stored in $model_path"
        weights = CSV.read(model_path, DataFrame)
        model = LinearModel(weights[!, :Weights])
    else
        if force_retrain
            @warn "Forcing re-training and serialization to $model_path from features in $features_path"
            rm(model_path, force=true)
        else
            @info "Serialized model not found at $model_path, let's try to train a new model from features in $features_path"
        end
        model = train_and_serialize_linear_model(features_path, model_path)
    end
    return model
end

function load_or_train_serialized_model_from_jls(model_path::String, features_path::String, force_retrain::Bool)::MachineModel
    
    if isfile(model_path) && !force_retrain
        @info "Found model stored in $model_path"
        model = load_model_from_jls(model_path)
    else
        if force_retrain
            @warn "Forcing re-training and serialization to $model_path from features in $features_path"
            rm(model_path, force=true)
        else
            @info "Serialized model not found at $model_path, let's try to train a new model from features in $features_path"
        end
        model = train_and_serialize_model(features_path, model_path; num_tuning_iterations = 100)
    end
    return model
end

function load_model_from_jls(model_path)::MachineModel
    return MachineModel(machine(model_path))
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

const features_to_exclude = Set([String(f) for f in [
    :CountHandWood,
    :CountHandBrick,
    :CountHandPasture,
    :CountHandStone,
    :CountHandGrain,
    :HasMostPoints,
    :NumberOfTurns,
    :CountVictoryPoint
]
])

function filter_bad_features!(df::DataFrame)
    bad_features = intersect(names(df), features_to_exclude)
    #@warn "Remove $bad_features from features !"
    #@debug "removing $feat from features while loading"
    select!(df, Not(bad_features))
end
function filter_bad_features(features::Vector)
    return filter(pair -> !(String(pair.first) in features_to_exclude), features)
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
function train_and_serialize_model(features_csv::String, output_path::String; num_tuning_iterations = 100)::DecisionModel
    tree = load_tree_model()
    tuned_mach = train_model_from_csv(tree, features_csv, num_tuning_iterations = num_tuning_iterations)
    @info "Serializing model trained on $features_csv into $output_path"
    MLJ.save(output_path, tuned_mach)
    return MachineModel(tuned_mach)
end

"""
    `train_and_serialize_linear_model(features_csv, output_path)`

This is the access point for re-training a model based on new features or engine bug fixes.
"""
function train_and_serialize_linear_model(features_csv::String, output_path::String; sv_threshold = 0.01)::LinearModel
    model = train_linear_model_from_csv(features_csv; sv_threshold = sv_threshold)
    @info "Serializing linear model trained on $features_csv into $output_path"
    df = DataFrame(Weights = model)
    CSV.write(output_path, df)
    return LinearModel(model)
end

function train_linear_model_from_csv(features_csv; sv_threshold = 0.01)
    df = CatanLearning.load_typed_features_from_csv(features_csv)
    y = [value == true ? 1.0 : 0.0 for value in df[:, :WonGame]]
    # Extract the feature matrix
    X = Matrix(df[:, Not(:WonGame)])
    pseudo_inv = pseudo_inverse(X, sv_threshold)
    model = pseudo_inv * y
    return model
end

function pseudo_inverse(X::Matrix, sv_threshold)
    (m,n) = size(X)
    U,S,Vt = svd(X)
    N_num = length(filter(x -> x >= sv_threshold, S))
    return transpose(Vt)[1:n, 1:N_num] * diagm(1 ./ S[1:N_num]) * transpose(U)[1:N_num, 1:m]
end

function add_perturbation!(model::MachineModel, magnitude)
    @warn "Not implemented - no perturbation added"
    return model
end

function add_perturbation!(model::LinearModel, magnitude)
    d = Normal(0.0, magnitude)
    model.weights += rand(d, size(model.weights))
end

function read_perturbed_linear_model(tourney_id, epoch_num, team, output_dir)::LinearModel
    file_name = "linear_model_$(tourney_id)_team_$(epoch_num).csv"
    model_path = "$output_dir/$file_name"
    weights = CSV.read(model_path, DataFrame)
    model = weights[!, :Weights]
    return LinearModel(model)
end
function write_perturbed_linear_model(tourney_path, epoch_num, team, model::LinearModel, models_dir, name = "model")
    df = DataFrame(Weights = model.weights)
    file_name = "$(epoch_num)_$(name)_$team.csv"
    output_path = joinpath(tourney_path, file_name)
    weights = CSV.write(output_path, df)
end