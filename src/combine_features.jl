using CSV
using DataFrames

folder_path = "features"

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
for file in csv_files
    println(file)
    df = CSV.read(joinpath(folder_path, file), DataFrame)
    append!(combined_df, df)
end

# Write the combined DataFrame to a new CSV file
CSV.write("data/features.csv", combined_df)
