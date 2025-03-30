#julia --project -e "using CatanLearning; CatanLearning.run(CatanLearning.Catan.DefaultRobotPlayer)" $((RANDOM % 1000))
julia --project ./src/CatanLearning.jl "$((RANDOM % 10000))"
