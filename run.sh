#parallel -N0 ./run_one.sh ::: {1..8}
time parallel --line-buffer -N0 ./run_one.sh ::: {1..8} 2>&1
