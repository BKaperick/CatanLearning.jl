function print_mutation(mutation::Dict)
    return join(["$c => $v" for (c,v) in mutation if v != 0], ", ")
end

function order_winners(unordered_winners)::Vector{Tuple{Union{Nothing,Symbol}, Int}}
    return [(k,v) for (k,v) in sort(collect(unordered_winners), by=x -> x.second, rev=true)]# if k !== nothing]
end
