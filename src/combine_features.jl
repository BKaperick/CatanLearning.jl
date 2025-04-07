using CSV
using DataFrames

folder_path = "new_features"

csv_files = readdir(folder_path)

for file in csv_files
    file_path = joinpath(folder_path, file)
    # Read the file contents
    lines = readlines(file_path)

    # Replace the first line using the regular expression
    if length(lines) > 0
        lines[1] = replace(lines[1], "\"WonGame\",\"WonGame\"" => "\"WonGame\"")
    end

    # Write the modified contents back to the file
    open(file_path, "w") do f
        for line in lines
            println(f, line)
        end
    end
end

combined_df = DataFrame()
old_df = DataFrame()

for file in csv_files
    println(file)
    df = CSV.read(joinpath(folder_path, file), DataFrame)
    if ("CountTotalSettlement" in names(df))
        append!(old_df, df)
        println("Skipping $file")
        continue
    end
    if "CountVictoryPoint_1" in names(df)
        select!(df, Not([:CountVictoryPoint_1]));
    end
    append!(combined_df, df)
end
#transform!(combined_df, [:CountSettlement, :CountCity] => (a,b) -> a+b => :CountTotalSettlement)
combined_df[!, :CountTotalSettlement] = combined_df[!, :CountSettlement] .+ combined_df[!, :CountCity]
append!(combined_df, old_df)

# Write the combined DataFrame to a new CSV file
CSV.write("data/new_features.csv", combined_df)
