using Random
using CSV
using MLJ
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles

function create_new_model(label, features_file)
    tree = get_tree()
    serialize_model_from_csv_features(tree, label, features_file)
end

function load_tree_model()
    #@load RandomForestClassifier pkg=BetaML verbosity=0
    @load RandomForestClassifier pkg=DecisionTree verbosity=0
end

"""
    try_load_model_from_csv(tree, model_file_name, features_file_name)

If the serialized file exists, then load it.  If not, train a new model and 
serialize it before returning it to caller.
"""
function try_load_model_from_csv(tree, model_file_name, features_file_name)::Machine
    @info "Looking for model stored in $model_file_name"
    if isfile(model_file_name)
        @info "Found model stored in $model_file_name"
        return load_model_from_csv(model_file_name)
    end
    @info "Not found, let's try to train a new model from features in $features_file_name"
    return serialize_model_from_csv_features(tree, features_file_name)
end
function load_model_from_csv(model_file_name)::Machine
    return machine(model_file_name)
end

function train_model_from_csv(tree, label::Symbol, csv_name="$(@__DIR__)../../features.csv")
    println("training from csv")
    # Load data
    data, header = readdlm(csv_name, ',', header=true)
    df = DataFrame(data, vec(header))
    coerce!(df, label => Multiclass{2})
    # Don't use label columns in training
    #select!(df, Not([:WonGame, :CountVictoryPoint]))
    select!(df, Not([:HasMostPoints]))
    df = DFM.@transform(df, :WonGame)

    # In production model, we don't partition, we use it all
    #df, df_test = partition(df, 0.7, rng=123)
    
    # Fit model machine to data
    y, X = unpack(df, ==(label));
    mach = machine(tree, X, y)
    Base.invokelatest(fit!, mach)

    return mach
end
function train_model_from_csv(tree, label::Symbol, features_csv="$(@__DIR__)../../features.csv")
    data, header = readdlm(features_csv, ',', header=true)
    df = DataFrame(data, vec(header))

    select!(df, Not([
        :CountHandWood,
        :CountHandBrick,
        :CountHandPasture,
        :CountHandStone,
        :CountHandGrain,
        :HasMostPoints,
        :CountVictoryPoint
        ]))

    coerce!(df, label => Multiclass{2})
    select!(df, Not([:HasMostPoints]))
    df = DFM.@transform(df, label)
    df, df_test = partition(df, 0.7, rng=123)

    y, X = unpack(df, ==(label));
    y_test, X_test = unpack(df_test, ==(label));


    thresholded_tree = BinaryThresholdPredictor(tree, threshold=0.5)
    ranges = [
        range(thresholded_tree, :threshold, lower=0.1, upper=0.9),
        range(thresholded_tree, :(model.min_purity_increase), lower=0.0, upper=0.9),
        range(thresholded_tree, :(model.min_samples_leaf), lower=4, upper=10),
        range(thresholded_tree, :(model.min_samples_split), lower=2, upper=8),
        #range(thresholded_tree, :(model.partial_sampling), lower=0.5, upper=0.9),
        range(thresholded_tree, :(model.n_trees), lower=5, upper=20)
    ]

    tuned_tree = TunedModel(
        thresholded_tree,
        tuning=RandomSearch(),
        resampling=CV(nfolds=6),
        range = ranges,
        measure = MatthewsCorrelation(),
        n=100
    )
    #mach = machine(tree, X, y) |> MLJ.fit!

    tuned_mach = machine(tuned_tree, X, y) |> MLJ.fit!
    return tuned_mach
end

function serialize_model_from_csv_features(tree, label, csv_name)
    mach = train_model_from_csv(tree, label, csv_name)
    MLJ.save("$(DATA_DIR)/model.jls", mach)
    return mach
end

function train_and_serialize_model(features_csv, label::Symbol, output_path)
    Tree = @load RandomForestClassifier pkg=DecisionTree verbosity=0
    tree = Base.invokelatest(Tree)
    tuned_mach = train_model_from_csv(tree, label, features_csv)
    MLJ.save(output_path, tuned_mach)
end
