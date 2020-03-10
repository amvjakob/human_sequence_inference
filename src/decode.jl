# decode.jl
using Random

include("updateRule.jl")


### Generate a sequence of length len containing elements 0 and 1

function generateSeq(len, seed = 1234)
    Random.seed!(seed)
    return rand([0, 1], len)
end


### Decode a sequence to compute transition probabilites

function decode(seq, m, alpha_0, rule::UpdateRule, callback = () -> nothing)
    len = length(seq)
    d = 2^m

    # check for correct dimensions
    @assert(size(alpha_0) == (2, d))

    # init update rule
    rule.init(alpha_0)

    # decode sequence
    for t in 1:len
        x_t = seq[t]

        if t < m + 1
            # we have some missing elements at the beginning of the window
            # we solve this by averaging over possible value
            minIdx = t > 1 ? seqToIdx(seq[1:t-1]) : 1
            step = 2^(t-1)
            maxIdx = d

            cols = minIdx:step:maxIdx
            for col in cols
                rule.update(x_t, col, 1.0 / length(cols))
            end
            callback(x_t, minIdx)
        else
            col = seqToIdx(seq[t-m:t-1])
            rule.update(x_t, col)
            callback(x_t, col)
        end
    end
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
