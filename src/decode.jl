# decode.jl
using Random

include("updateRule.jl")

### Generate a sequence of length len containing elements 0 and 1

function generateSeq(len, seed = 1234)
    Random.seed!(seed)
    return rand([0, 1], len)
end


### Decode a sequence to compute transition probabilites

# flag that indicates the action to take when the previous
# sequence is not yet m elements long
# ignore or average over possible previous sequences
function decode(seq, m, alpha_0, rule::UpdateRule;
    callback::Callback{F,R} = Callback((_...) -> nothing, Nothing),
    ignoreFirstM = true) where {F,R}

    len = length(seq)
    d = 2^m

    # check for correct dimensions
    @assert(size(alpha_0) == (2, d))

    # init update rule
    rule.init(alpha_0)

    # callback result
    callbackResult = Array{R,1}(undef, len+1)

    # partial window
    if !ignoreFirstM
        for t in 1:m
            x_t = seq[t]

            # we have some missing elements at the beginning of the window
            # we solve this by averaging over possible value
            minIdx = t > 1 ? seqToIdx(seq[1:t-1]) : 1
            step = 2^(t-1)
            maxIdx = d

            # perform callback
            callbackResult[t] = callback.run(rule, x_t, minIdx)

            # add pseudocount of 1 to every possible value
            cols = minIdx:step:maxIdx
            for col in cols
                rule.update(x_t, col)
            end
        end
    end

    # full window
    for t in m+1:len
        x_t = seq[t]
        col = seqToIdx(seq[t-m:t-1])

        callbackResult[t] = callback.run(rule, x_t, col)
        rule.update(x_t, col)
    end

    # add final callback result
    x_t = seq[end]
    col = seqToIdx(seq[end-m:end-1])
    callbackResult[end] = callback.run(rule, x_t, col)

    return callbackResult
end

function seqToIdx(seq)
    # concat sequence to string
    str = join(seq)
    return str == "" ? 1 : (parse(Int64, str, base = 2) + 1)
end

function idxToSeq(idx)
    # returns a string
    str = string(idx - 1, base = 2)
end
