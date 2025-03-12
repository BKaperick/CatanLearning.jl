function build_roc(results, y_true, res = 0.1)
    #classes = Set(vcat(collect(keys(results))...))
    classes = [0,1]
    data = []
    for thresh = 0:res:1
        tp,fp,tn,fn = 0,0,0,0
        for (res, act) in zip(results, y_true)
            if ~haskey(res, 0)
                res[0] = 0
            end
            if ~haskey(res, 1)
                res[1] = 0
            end
            if res[0] >= thresh
                if act == 0
                    tn += 1
                else
                    fn += 1
                end
            else
                if act == 1
                    tp += 1
                else
                    fp += 1
                end
            end
        end
        tpr = tp / (tp + fn) 
        fpr = fp / (fp + tn)
        push!(data, (thresh, tpr, fpr))
    end
    return data
end
