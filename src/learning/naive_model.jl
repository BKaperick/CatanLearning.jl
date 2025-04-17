function naive_model_proba(X_in)
    X = X_in[1]
    #println("analyzing $X")
    unscaled_feature = X[!,:CountSettlement] + X[!,:CountCity] + 2*(X[!,:HasLargestArmy] + X[!,:HasLongestRoad])
    return min.(unscaled_feature ./ 10, 1.0)
end

MMI.@mlj_model mutable struct NaivePublicVictoryPoints <: MLJModelInterface.Deterministic
end

MMI.fit(::NaivePublicVictoryPoints, verbosity, data...) = (nothing, nothing, nothing)
MMI.predict(model::NaivePublicVictoryPoints, fitresult, new_data...) = naive_model_proba(new_data)

#model = NaivePublicVictoryPoints()
#naive_machine = machine(model, nothing, nothing) |> MLJ.fit!
