# utils.jl

using Dates, SpecialFunctions

### log to console and immediately show
function lg(x...)
  println("[", now(), "] ", join(x, " ")...)
  flush(stdout)
end


### ln range between x1 and x2
# sequence might not be n elements long
function lnrange(x1::Integer, x2::Integer, n::Integer)
  return unique(round(Int, Base.MathConstants.e^y) for y in range(x1, x2, length=n))
end

### log2 range between x1 and x2
# sequence will be n elements long
function log2range(x1::Integer, x2::Integer, n::Integer)
  N = n
  rng(s,e,l) = unique(round(Int, 2^y) for y in range(x1, x2, length=n))
  while length(rng(x1,x2,n)) < N
    n += 1
  end
  return rng(x1,x2,n)
end

### log10 range between x1 and x2
# sequence might not be n elements long
function logrange(x1::Integer, x2::Integer, n::Integer)
  return unique(round(Int, 10^y) for y in range(x1, x2, length=n))
end


### unwraps an array of arrays to a multidimensional array
# default (return input array)
function unwrap(a::Array{T,N}) where {T <: Real, N}
  return a
end

function unwrap(a::Array{Array{T,N},1}) where {T <: Real, N}
  shape = pushfirst!(collect(1:N), N+1) 
  return permutedims(cat(a..., dims=N+1), shape)
end

function unwrap(a::Array{Array{T,1},2}) where T <: Real
  result = zeros(size(a)..., length(a[1]))
  for i in CartesianIndices(a)
    result[i,:] = a[i]
  end
  return result
end

# TODO: rename to unwrap?
# unwrap an array of arrays of arrays
function unwrap_outer(a::Array{Array{T,N},1}) where {T <: AbstractArray, N}
  shape = pushfirst!(collect(1:N), N+1) 
  return permutedims(cat(a..., dims=N+1), shape)
end

### compute the surprise modulation factor gamma
function compute_gamma(surprise::Float64, m::Real)
  return m * surprise / (1 + m * surprise)
end


### compute the expected probability of x for a Dirichlet distribution
# alpha: Dirichlet distribution parameters, array of length N
# x:     observation for which to compute E[theta]
function compute_theta(x::Integer, alpha::Array{Float64,1})
  # validity checks
  @assert(length(alpha) > 1)
  @assert(0 <= x < length(alpha))

  # x+1 because arrays are 1-indexed
  return alpha[x + 1] / sum(alpha)
end


### compute Bayes Factor surprise for observation x under current belief
# x:       current observation, integer ∈ [0, N-1]
# alpha_0: initial belief, array of length N
# alpha_t: current belief, array of length N
function compute_sbf(x::Integer,
                    alpha_0::Array{Float64,1},
                    alpha_t::Array{Float64,1})
  # validity checks
  @assert(length(alpha_0) == length(alpha_t))
  @assert(0 <= x < length(alpha_t))

  # Bayes Factor is ratio of probabilities
  return compute_theta(x, alpha_0) / compute_theta(x, alpha_t)
end

# TODO: needed?
# Bayes Factor surprise for particle filtering
# alpha_t: current belief, array of length P, each containing an array of length N
# w_t:     particle weights, array of length P
function compute_sbf(x::Integer,
                    alpha_0::Array{Float64,1},
                    alpha_t::Array{Array{Float64,1},1},
                    w_t::Array{Float64,1})
  # validity checks
  @assert(all(length.(alpha_t) .== length(alpha_0)))
  @assert(length(w_t) == length(alpha_t))
  @assert(0 <= x < length(alpha_t))

  # compute theta under alpha_t
  p_t = sum(compute_theta.(x, alpha_t) .* w_t)

  # surprise is ratio of probabilities
  return compute_theta(x, alpha_0) / p_t
end


### compute weighted harmonic mean of an array
# array:   array of length L
# weights: array of length L
function weighted_harmonic_mean(array, weights)
  s = 0.0
  n = length(array)
  @assert(length(weights) == n)

  for i in 1:n
    @inbounds s += weights[i] * inv(array[i])
  end

  return sum(weights) / s
end


### find nonzero elements in array
function findnz(array)
  a = similar(array, Int)
  count = 1
  @inbounds for i in eachindex(array)
    a[count] = i
    count += (array[i] != zero(eltype(array)))
  end
  return resize!(a, count-1)
end


### find non-NaN elements in array
function findnonnan(array)
  a = similar(array, Int)
  count = 1
  @inbounds for i in eachindex(array)
    a[count] = i
    count += !isnan(array[i])
  end
  return resize!(a, count-1)
end


### multivariate Beta function
function ln_beta_fn(alpha::Array{Float64,1})
  return sum(loggamma.(alpha)) - loggamma(sum(alpha))
end

function beta_fn(alpha::Array{Float64,1})
  return exp(ln_beta_fn(alpha))
end


### dirichlet distribution
# theta: array of length L
# alpha: array of length L
function dirichlet_fn(theta::Array{Float64,1}, alpha::Array{Float64,1})
  @assert(size(theta) == size(alpha))
  @assert(sum(theta) == 1.0)
  return 1.0 / beta_fn(alpha) * prod(theta .^ (alpha .- 1))
end


### create an array of zeros with a 1 at the given position
function oneatpos(pos, shape)
  arr = zeros(shape)
  arr[pos...] = 1
  return arr
end
