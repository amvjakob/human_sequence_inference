using GLM, Printf, Dates, JuliennedArrays

include("decode.jl")
include("utils.jl")
include("updateRule.jl")

function computeShannonSurprise(rule, x_t, col)
    theta = rule.computeTheta(col)
    theta = x_t == 1 ? theta : 1 - theta
    return -log(theta) / log(2.0)
end

function decodeShannonSurprise(seq, m, alpha_0, rule, ignoreFirstM = true)
    # define callback to compute Shannon surprises
    callback = Callback(computeShannonSurprise, Float64)

    # decode sequence
    surprises = decode(seq, m, alpha_0, rule;
                       callback = callback, ignoreFirstM = ignoreFirstM)

    if ignoreFirstM && m > 0
        surprises[1:m] .= -log(0.5) / log(2.0)
    end

    return surprises # [2:end]
end


function onehotencode(shape, idx)
    onehot = zeros(shape...)
    onehot[:,idx] .= 1
    return onehot
end

function onehotencodeSurprises(s::Tuple{Int,Array{Float64,1}})
    block, surprise = s

    onehotShape = (length(surprise), 4)
    onehot = onehotencode(onehotShape, block)

    return hcat(onehot, surprise)
end

function regression(subject, sensor, time, rule, m, alpha_0)
    # get suprises per block
    surprises = decodeShannonSurprise.(subject.seq, m, Ref(alpha_0), Ref(rule))
    surprises = map((s,i) -> s[i], surprises, subject.seqIdx)

    # one-hot encode surprises
    surprisesOneHot = map(onehotencodeSurprises, enumerate(surprises))
    #surprisesOneHot = map(v -> hcat(ones(size(v)...), v), surprises)

    # get megs per block
    megs = map(m -> m[:,sensor,time], subject.meg)

    # calc regression
    X = reduce(vcat, surprisesOneHot)
    y = reduce(vcat, megs)

    # normalize y
    my = mean(y)
    sy = std(y)
    y = (y .- my) ./ sy

    ols = fit(LinearModel, X, y);

    megs = map(m -> (m .- my) ./ sy, megs)
    return surprises, megs, ols, y
end

function mapSubject(subject, models, keepBlockLevel = false, verbose = 2)
    # init rule decode results
    surprises = Array{Array{Array{Float64,2},1},1}(undef, length(models))

    # decode sequence for every rule
    if verbose > 1
        lg("mapSubject: decoding sequence for all rules")
    end

    for (m, model) in enumerate(models)
        modelSurprises = Array{Array{Float64,2},1}(undef, length(model))
        
        for (r, rule) in enumerate(model)
            # decode sequence            
            values = decodeShannonSurprise.(subject.seq, rule["m"], Ref(rule["alpha_0"]), Ref(rule["rule"]))
            # map decoded values to overlapping index
            values = map((s, idx) -> s[idx], values, subject.seqIdx)

            if keepBlockLevel
                # one-hot encode values by block
                values = map(onehotencodeSurprises, enumerate(values))
            else
                # hcat to make 2-dimensional
                values = map(v -> hcat(ones(size(v)...), v), values)
            end

            # remove block-level structure
            @inbounds modelSurprises[r] = reduce(vcat, values)
        end
        
        surprises[m] = modelSurprises
    end

    # get MEG values
    if verbose > 1
        lg("mapSubject: decoding MEG")
    end
    megs = reduce(vcat, subject.meg)

    return surprises, megs
end
;
