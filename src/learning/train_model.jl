using Random
using CSV
using MLJ
using MLJDecisionTreeInterface
using DecisionTree
using DelimitedFiles
using DataFrames
import DataFramesMeta as DFM
#df = DataFrame(CSV.File(ARGS[1]))
#df = DataFrame("../../features.csv")
using Logging

#logger = ConsoleLogger(stderr, Logging.Debug)
#logger = ConsoleLogger(stderr, Logging.Info)
#logger = ConsoleLogger(stderr, Logging.LogLevel(5000))
#global_logger(logger)

#features_csv = "./data/features.csv"

features_csv = "../../data/features.csv"

