# inference.jl

### inference over m (window length)

# priors over m
function prior_fixed(m)
  prior = zeros(m + 1)
  prior[m + 1] = 1
  return prior
end

function prior_uniform(mmax)
  return ones(mmax + 1) / (mmax + 1)
end

function prior_geometric_truncated(q, mmax = 99)
  @assert(0 < q <= 1)
  return (1 - q) / (1 - q ^ (mmax + 1)) * q .^ collect(0:mmax)
end