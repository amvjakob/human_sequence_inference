# main.jl
include("decode.jl")
include("updateRule.jl")

len = 1000
m = 5
alpha_0 = ones(2, 2^m)
rule = perfect()
#rule = leaky(10)
#rule = varSMiLe(3.6)
#rule = particleFiltering(0.2, 10, 2)

seq = generateSeq(len)
decode(seq, m, alpha_0, rule)

# show result
@show(rule.params())
