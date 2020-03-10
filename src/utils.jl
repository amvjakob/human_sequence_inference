# utils.jl
using SpecialFunctions


### compute the modulation factor gamma

function computeGamma(surprise, m)
    return m * surprise / (1.0 + m * surprise)
end


### compute "Base Factor surprise" for observation x_t

# x_t: current observation
# alpha_0: shape should be (2,) (binary signal)
# alpha_t: shape should be (2,) (binary signal)
function computeSBF(x_t, alpha_0, alpha_t)
    # check for same size
    @assert(size(alpha_0) == size(alpha_t))

    # probability under given state
    # add 1 to x_t to go from value (0, 1) to index (1, 2)
    p = (alpha) -> alpha[x_t + 1] / sum(alpha)

    # surprise is ratio of probabilities
    return p(alpha_0) / p(alpha_t)
end

function computeSBFFromChi(x_t, chi_0, chi_t)
    return computeSBF(x_t, chi_0 .+ 1.0, chi_t .+ 1.0)
end

# x_t: current observation
# alpha_0: shape should be (2,) (binary signal)
# alpha_t: shape should be (N, 2,) (binary signal)
# w_t: shape should be (N,) (binary signal)
function computeSBF(x_t, alpha_0, alpha_t, w_t)
    # probability under given state
    # add 1 to x_t to go from value (0, 1) to index (1, 2)
    p = (alpha) -> alpha[x_t + 1] / sum(alpha)
    
    p_t = 0
    for i in 1:length(w_t)
       p_t += w_t[i] * p(alpha_t[i,:]) 
    end

    # surprise is ratio of probabilities
    return p(alpha_0) / p_t
end


### utility functions to switch between alpha and chi

function chiToAlpha(chi)
    return deepcopy(chi) .+ 1.0
end

function alphaToChi(alpha)
    return deepcopy(alpha) .- 1.0
end


### compute weighted harmonic mean

function weightedHarmonicMean(arr, w)
    s = 0.0
    n = length(arr)
    @assert(length(w) == n)

    for i in 1:n
        @inbounds s += w[i] * inv(arr[i])
    end
    return sum(w) / s
end


### multivariate Beta function

# alpha: shape should be (*,)
function betaFn(alpha)
    return prod(gamma.(alpha)) / gamma(sum(alpha))
end


### dirichlet distribution

# theta: shape should be (*,)
# alpha: shape should be (*,)
function dirichletFn(theta, alpha)
    @assert(size(theta) == size(alpha))
    @assert(sum(theta) == 1.0)
    return 1.0 / betaFn(alpha) * prod(theta .^ (alpha .- 1))
end


### create an array of zeros with a 1 at the given position

function oneAtPos(pos, shape)
    arr = zeros(shape)
    arr[pos...] = 1
    return arr
end
