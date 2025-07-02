using CSV
using DataFrames
using Plots

# Read the CSV file
df = CSV.read("../RunScripts/data/state_values.csv", DataFrame, header=["state_key", "value"])

# Plot the "Temperature" column
sorted_df = sort(df, :value)
plot(sorted_df[!,:value], title="Estimated value after 9000 games", ylabel="value", legend=false)

savefig("./data/sorted_value_estimates.png")
