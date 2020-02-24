# decode.jl
using Random

include("updateRule.jl")

### Generate a sequence of length len containing elements 0 and 1
function generateSeq(len, seed = 1234)
    Random.seed!(seed)
    return rand([0, 1], len)
end

### Decode a sequence to compute transition probabilites
function decode(seq, m, alpha0, rule::UpdateRule)
    len = length(seq)
    d = 2^m

    # check for correct dimensions
    @assert(size(alpha0) == (2, d))
    alpha = ones(2, 2^m, len)
    alpha[:,:,1] = alpha0

    # decode sequence
    transitions = zeros(2, d, len)
    for t in 1:len
        if t < m + 1
            # we have some missing elements at the beginning of the window
            # we solve this by averaging over possible value
            minIdx = t > 1 ? seqToIdx(seq[1:t-1]) : 1
            step = 2^(t-1)
            maxIdx = d

            indices = minIdx:step:maxIdx
            transitions[seq[t]+1, indices, t] .+= 1.0 / length(indices)
        else
            # add transition to state x_t
            idx = seqToIdx(seq[t-m:t-1])
            transitions[seq[t]+1, idx, t] += 1

            # update alpha
            alpha[:,:,t] = rule.update(t, transitions[:,:,1:t], alpha0)
        end
    end

    return alpha
end

function seqToIdx(seq)
    # concat sequence to string
    str = join(seq)
    return parse(Int64, str, base = 2) + 1
end

function idxToSeq(idx)
    # returns a string
    str = string(idx - 1, base = 2)
end
