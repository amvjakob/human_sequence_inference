# main.jl
using Distributions

include("decode.jl")
include("updateRule.jl")

len = 100
m = 5
alpha0 = ones(2, 2^5)
rule = perfect() # leaky(w), varSMiLe(m)

seq = generateSeq(len)
alphaDecoded = decode(seq, m, alpha0, rule)

# show result
alphaDecoded[:,:,end]

# dirichlet
# dir = Distributions.Dirichlet(alphaDecoded[:,:,end])
