# plot.jl
using PyPlot

include("decode.jl")
include("utils.jl")
include("updateRule.jl")

function plotSurprise(rule)
    len = 1000
    m = 1
    alpha_0 = ones(2, 2^m)

    # generate sequence { 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 ... }
    seq = Array{Int32}(undef, len)
    for i = 1:len/2
        seq[i] = i % 3 == 0
    end

    for i = len/2:len
        seq[i] = i % 4 == 0
    end

    rule = perfect()

    surprises = zeros(len)
    surprisesIdx = 1
    function callback(x_t, col)
        surprises[surprisesIdx] = computeSBF(x_t, alpha_0[:,col], rule.params()[:,col])
        surprisesIdx += 1
    end

    decode(seq, m, alpha_0, rule, callback)

    x = range(1, len, length=len)

    clf()
    figure(1)
    display(plot(range(1, len, length=len), surprises))
    xlabel("t")
    ylabel("S_{BF}")
    title("Surprise")
    grid("on")
    gcf()

end

#default(legend = false)
#plotSurprise(perfect())
