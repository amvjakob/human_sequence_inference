# utils.jl
using SpecialFunctions, Pipe, JuliennedArrays

function lg(x...)
    println("[", now(), "] ", join(x, " ")...)
    flush(stdout)
end

function lnrange(x1::Int, x2::Int, n::Int)
    return unique(round(Int, Base.MathConstants.e^y) for y in range(x1, x2, length=n))
end

function log2range(x1::Int, x2::Int, n::Int)
    N = n
    rng(s,e,l) = unique(round(Int, 2^y) for y in range(x1, x2, length=n))
    while length(rng(x1,x2,n)) < N
        n += 1
    end
    return rng(x1,x2,n)
end

function logrange(x1::Int, x2::Int, n::Int)
    return unique(round(Int, 10^y) for y in range(x1, x2, length=n))
end

function arrayOfArrayToMatrix(array::Array{T,N}) where {T <: Real,N}
    return array
end

function arrayOfArrayToMatrix(array::Array{Array{T,1},1}) where T <: Real
    return @pipe cat(array..., dims=2) |> permutedims(_, (2, 1))
end

function arrayOfArrayToMatrix(array::Array{Array{T,1},2}) where T <: Real
    return @pipe cat(array..., dims=2) |> reshape(_, size(array[1])..., size(array)...)
end

function arrayOfArrayToMatrix(array::Array{Array{T,2},1}) where T <: Real
    return @pipe cat(array..., dims=3) |> permutedims(_, (3, 1, 2))
end

function arrayOfArrayToMatrix(array::Array{Array{T,3},1}) where T <: Real
    return @pipe cat(array..., dims=4) |> permutedims(_, (4, 1, 2, 3))
end


### compute the modulation factor gamma

function computeGamma(surprise::Float64, m::Float64)
    return m * surprise / (1.0 + m * surprise)
end


### compute theta (P(y = x_t)) from alpha

function computeTheta(alpha::Array{Float64,1}, x_t = 1)
    # might have to adapt this code for non-binary signals
    @assert(length(alpha) == 2)
    @assert(0 <= x_t <= 1)

    return alpha[x_t + 1] / sum(alpha)
end

utilsComputeTheta = computeTheta

### compute "Base Factor surprise" for observation x_t

# x_t: current observation
# alpha_0: shoule be of length 2 (binary signal)
# alpha_t: should be of length 2 (binary signal)

function computeSBF(x_t::Int, alpha_0::Array{Float64,1}, alpha_t::Array{Float64,1})
    @assert(size(alpha_0) == size(alpha_t))

    # surprise is ratio of probabilities
    return computeTheta(alpha_0, x_t) / computeTheta(alpha_t, x_t)
end

utilsComputeSBF = computeSBF

function computeSBFFromChi(x_t::Int, chi_0::Array{Float64,1}, chi_t::Array{Float64,1})
    return computeSBF(x_t, chi_0 .+ 1, chi_t .+ 1)
end

utilsComputeSBFFromChi = computeSBFFromChi


# alpha_0: should be of length 2 (binary signal)
# alpha_t: should be N x 2 (binary signal)
# w_t: should be of length N

function computeSBF(x_t::Int,
                    alpha_0::Array{Float64,1},
                    alpha_t::Array{Array{Float64,1},1},
                    w_t::Array{Float64,1})

    # check for same number of particles
    @assert(length(w_t) == size(alpha_t, 1))

    # compute theta under alpha_t
    p_t = sum(computeTheta.(alpha_t, x_t) .* w_t)

    # surprise is ratio of probabilities
    return computeTheta(alpha_0, x_t) / p_t
end

utilsComputeSBF = computeSBF

### utility functions to switch between alpha and chi

function chiToAlpha(chi)
    return chi .+ 1.0
end

function alphaToChi(alpha)
    return alpha .- 1.0
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


### find nonzero elements in array

function findnz(c)
    a = similar(c, Int)
    count = 1
    @inbounds for i in eachindex(c)
        a[count] = i
        count += (c[i] != zero(eltype(c)))
    end
    return resize!(a, count-1)
end

function findnonnan(c)
    a = similar(c, Int)
    count = 1
    @inbounds for i in eachindex(c)
        a[count] = i
        count += !isnan(c[i])
    end
    return resize!(a, count-1)
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
