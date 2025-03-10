using Random
using CSV
using DataFrames
import DataFramesMeta as DFM
using DelimitedFiles
using BetaML
import BetaML
using Logging
using MLJScikitLearnInterface
using MLJ
using Plots

logger = ConsoleLogger(stderr, Logging.Debug)
global_logger(logger)


# Read in data
features_csv = "./data/features.csv"
data, header = readdlm(features_csv, ',', header=true)
# TODO remove this when ready for real training
#data = data[1:10_000,1:end]

# Set up training and test partitions
df = DataFrame(data, vec(header))
transform!(df, :WonGame => ByRow(Int8) => :WonGame)
#coerce!(df, :WonGame => Multiclass{2})
#df = DFM.@transform(df, :WonGame)
#df, df_test = partition(df, 0.7, rng=123)

sample = randsubseq(1:size(df,1), 0.7)
df_train = df[sample, :]
notsample = [i for i in 1:size(df,1) if isempty(searchsorted(sample, i))]
df_test = df[notsample, :]

y, df_X = unpack(df_train, ==(:WonGame));
X = Matrix(df_X)
y_test, df_X_test = unpack(df_test, ==(:WonGame));
X_test = Matrix(df_X_test)

model = BetaML.Trees.RandomForestEstimator(
    n_trees = 30,
    max_depth = nothing,
    min_gain = 0.0,
    min_records = 2,
    max_features = nothing,
    sampling_share = 1.0,
    force_classification = true,
    splitting_criterion = BetaML.Utils.gini,
    beta = 0.0,
    oob = true#,
    #tune_method = BetaML.Utils.SuccessiveHalvingSearch
)

y_hat_train = BetaML.fit!(model, X, y) |> BetaML.mode
#y_hat_train_2   = BetaML.predict(model, X)

y_hat_test   = BetaML.predict(model, X_test)
acc_train, acc_test  = BetaML.accuracy.([y,y_test],[y_hat_train,y_hat_test])
results = DataFrame(model=String[],train_acc=Float64[],test_acc=Float64[])
push!(results,["RF",acc_train, acc_test])

cfm = BetaML.ConfusionMatrix(categories_names=Dict(1=>"Win",0=>"Loss"))
BetaML.fit!(cfm,y_test, y_hat_test) # the output is by default the confusion matrix in relative terms
print(cfm)

res = BetaML.info(cfm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")
