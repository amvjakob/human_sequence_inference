# main.jl
include("decode.jl")
include("updateRule.jl")

len = 1000
m = 5
alpha0 = ones(2, 2^m)
#rule = perfect()
#rule = leaky(200)
rule = varSMiLe(0.1)
#rule = particleFiltering(5, 10, 10)

seq = generateSeq(len)
alphaDecoded = decode(seq, m, alpha0, rule)

# show result
@show(alphaDecoded[:,:,end])

# dirichlet
# dir = Distributions.Dirichlet(alphaDecoded[:,:,end])
