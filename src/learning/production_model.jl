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

function try_load_model_from_csv(team::Symbol, configs::Dict)::Machine
    key = "MODEL"
    cached = get_ml_cache_config(configs, team, key)::Union{Machine, Nothing}
    if cached !== nothing
        return cached
    end
    machine = try_load_serialized_model_from_csv(get_player_config(configs, key, team),  get_player_config(configs, "FEATURES", team))
    update_ml_cache!(configs, team, key, machine)
    return machine
end

function try_load_linear_model_from_csv(team::Symbol, configs::Dict)::Vector{Float64}
    key = "MODEL"
    cached = get_ml_cache_config(configs, team, key)::Union{Vector{Float64}, Nothing}
    if cached !== nothing
        return cached
    end
    model_path = get_player_config(configs, key, team)
    if isfile(model_path)
        @info "Found $key model stored in $model_path"
        weights = CSV.read(model_path, DataFrame)
        model = weights[!, :Weights]
    else
        features_path = get_player_config(configs, "FEATURES", team)
        @info "$model model not found, let's try to train a new model from features in $features_path"
        model = train_and_serialize_linear_model(features_path, model_path)
    end
    update_ml_cache!(configs, team, key, model)
    return model
end

function try_load_public_model_from_csv(team::Symbol, configs::Dict)::Machine
    key = "PUBLIC_MODEL"
    cached = get_ml_cache_config(configs, team, key)#::Union{Vector{Float64}, Nothing}
    if cached !== nothing
        return cached
    end
    machine = try_load_serialized_model_from_csv(get_player_config(configs, key, team),  get_player_config(configs, "PUBLIC_FEATURES", team))
    update_ml_cache!(configs, team, key, machine)
    return machine
end

"""
    try_load_model_from_csv(tree, model_file_name, features_file_name)

If the serialized file exists, then load it.  If not, train a new model and 
serialize it before returning it to caller.
"""
function try_load_serialized_model_from_csv(model_file_name::String, features_file_name::String)::Machine
    
    if isfile(model_file_name)
        @info "Found model stored in $model_file_name"
        return load_model_from_csv(model_file_name)
    end
    @info "Model not found, let's try to train a new model from features in $features_file_name"
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
    return tuned_mach
end

"""
    `train_and_serialize_linear_model(features_csv, output_path)`

This is the access point for re-training a model based on new features or engine bug fixes.
"""
function train_and_serialize_linear_model(features_csv::String, output_path::String; sv_threshold = 0.01)::Vector{Float64}
    model = train_linear_model_from_csv(features_csv; sv_threshold = sv_threshold)
    @info "Serializing linear model trained on $features_csv into $output_path"
    df = DataFrame(Weights = model)
    CSV.write(output_path, df)
    return model
end

function train_linear_model_from_csv(features_csv; sv_threshold = 0.01)
    df = CatanLearning.load_typed_features_from_csv(features_csv)
    y = [value == true ? 1.0 : 0.0 for value in df[:, :WonGame]]
    # Extract the feature matrix
    X = Matrix(df[:, Not(:WonGame)])
    (m,n) = size(X)
    U,S,Vt = svd(X)
    N_num = length(filter(x -> x >= sv_threshold, S))
    pseudo_inv = transpose(Vt)[1:n, 1:N_num] * diagm(1 ./ S[1:N_num]) * transpose(U)[1:N_num, 1:m]
    model = pseudo_inv * y
    return model
end

function get_perturbed(model::LinearModel)::LinearModel
    new_weights = copy(model.weights)

    d = Normal(0.0, 0.1)
    new_weights += rand(d, size(new_weights))
    return LinearModel(new_weights)
end
function add_perturbation!(model::LinearModel)
    d = Normal(0.0, 0.1)
    model.weights += rand(d, size(model.weights))
end
function add_perturbation!(model::MachineModel)
    # TODO implement later if needed
end

function read_perturbed_linear_model(tourney_id, epoch_num, team, output_dir)::LinearModel
    file_name = "linear_model_$(tourney_id)_team_$(epoch_num).csv"
    model_path = "$output_dir/$file_name"
    weights = CSV.read(model_path, DataFrame)
    model = weights[!, :Weights]
    return LinearModel(model)
end
function write_perturbed_linear_model(tourney_path, epoch_num, team, model::LinearModel, models_dir)
    df = DataFrame(Weights = model.weights)
    file_name = "$(epoch_num)_$team.csv"
    output_path = joinpath(tourney_path, file_name)
    weights = CSV.write(output_path, df)
end