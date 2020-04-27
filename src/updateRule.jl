# updateRule.jl
using Distributions, Pipe, JuliennedArrays

include("utils.jl")


### callback skeleton

mutable struct Callback{F,R}
    run::F

    function Callback(fun::F, R) where F
        return new{F,R}(fun)
    end
end


### update rule skeleton

mutable struct UpdateRule{I,U,P,T,S}
    # flag that indicates whether to update all parameters
    # or just the current column
    updateAll::Bool

    # function that initializes inital parameters
    init::I

     # function that updates the current parameters
    update::U

    # function that returns the current parameters
    params::P

    # function that computes theta
    computeTheta::T

    # function that computes the Surprise Bayes Factor
    computeSBF::S

    # str
    str
end


### utility functions

function build_rules_leaky(ws)
    return map(w -> leaky(w, true), ws)
end

function build_rules_varsmile(ms)
    return map(m -> varSMiLe(m, true), ms)
end

function build_models(rules::Array{UpdateRule{I,U,P,T,S},1},
        ms::Array{Int,1},
        names::Array{Str,1}) where {I,U,P,T,S,Str <: AbstractString}
    
    return map(m -> map(r -> Dict(
                "m" => m[1],
                "alpha_0" => ones(2,2^m[1]),
                "rule" => r,
                "name" => m[2]), rules), zip(ms, names))
end

function build_models(rules::Array{UpdateRule,1},
        ms::Array{Int,1},
        names::Array{Str,1}) where {Str <: AbstractString}
    
    return map(m -> map(r -> Dict(
                "m" => m[1],
                "alpha_0" => ones(2,2^m[1]),
                "rule" => r,
                "name" => m[2]), rules), zip(ms, names))
end

function build_models(rules::Array{UpdateRule,1}, ms::Array{Int,1})
    
    return build_models(rules, ms, map(m -> latexstring("m = $m"), ms))
end

function build_models(rules::Array{UpdateRule{I,U,P,T,S},1},
        ms::Array{Int,1}) where {I,U,P,T,S}
    
    return build_models(rules, ms, map(m -> latexstring("m = $m"), ms))
end



### perfect integration
function perfect()

    # set inital state
    alpha = Array{Float64,2}(undef, 0, 0)

    # init state
    function init(alpha_0::Array{Float64,2})
        alpha = copy(alpha_0)
    end

    # update state
    function update(x_t::Int, col::Int)
        alpha[x_t+1, col] += 1
    end

    # get state
    function params()
        return alpha
    end

    # compute theta
    function computeTheta(col::Int)
        return utilsComputeTheta(alpha[:,col])
    end

    # compute surprise
    function computeSBF(x_t::Int, col::Int, alpha_0::Array{Float64,1})
        return utilsComputeSBF(x_t, alpha_0, alpha[:,col])
    end

    return UpdateRule(
        false,
        init,
        update,
        params,
        computeTheta,
        computeSBF,
        "perfect()"
    )
end


### leaky integration

# w: leak factor
function leaky(w, updateAll = false)

    if w == Inf
        return perfect()
    end

    # set inital state
    alpha  = Array{Float64,2}(undef, 0, 0)
    alpha0 = Array{Float64,2}(undef, 0, 0)
    decay = exp(-1.0 / w)

    # init state
    function init(alpha_0::Array{Float64,2})
        alpha  = copy(alpha_0)
        alpha0 = copy(alpha_0)
    end

    # update state
    function updateCol(x_t::Int, col::Int)
        alpha[x_t+1,col] += 1
        alpha[:,col] = decay * (alpha[:,col] - alpha0[:,col]) + alpha0[:,col]
    end

    function updateAllCols(x_t::Int, col::Int)
        alpha[x_t+1,col] += 1
        alpha = decay * (alpha - alpha0) + alpha0
    end

    update = updateAll ? updateAllCols : updateCol;

    # get state
    function params()
        return alpha
    end

    # compute theta
    function computeTheta(col::Int)::Float64
        return utilsComputeTheta(alpha[:,col])
    end

    # compute surprise
    function computeSBF(x_t::Int, col::Int, alpha_0::Array{Float64,1})
        return utilsComputeSBF(x_t, alpha_0, alpha[:,col])
    end

    return UpdateRule(
        updateAll,
        init,
        update,
        params,
        computeTheta,
        computeSBF,
        "leaky($w, $updateAll)"
    )
end


### variational SMiLe

# m: p_c / (1 - p_c), where p_c = probability of change
function varSMiLe(m, updateAll = false)

    # set initial state
    chi_0 = Array{Float64,2}(undef, 0, 0)
    chi_t = Array{Float64,2}(undef, 0, 0)

    # init state
    function init(alpha_0::Array{Float64,2})
        # chi is 2 x 2^m, m = window length
        chi_0 = alphaToChi(alpha_0) |> copy
        chi_t = alphaToChi(alpha_0) |> copy
    end


    # update state
    function updateCol(gamma::Float64, col::Int)
        chi_t[:,col] = (1.0 - gamma) .* chi_t[:,col] + gamma .* chi_0[:,col]
    end

    function updateAllCols(gamma::Float64, col::Int)
        chi_t = (1.0 - gamma) .* chi_t + gamma .* chi_0
    end

    updateChi = updateAll ? updateAllCols : updateCol

    function update(x_t::Int, col::Int)
        # compute gamma
        sbf = computeSBFFromChi(x_t, chi_0[:,col], chi_t[:,col])
        gamma = computeGamma(sbf, m)

        # update chi
        updateChi(gamma, col)

        # add observation
        chi_t[x_t+1,col] += 1
    end

    # get state
    function params()
        # convert to corresponding alpha
        return chiToAlpha(chi_t) |> copy
    end

    # compute theta
    function computeTheta(col::Int)
        alpha = params()
        return utilsComputeTheta(alpha[:,col])
    end

    # compute surprise
    function computeSBF(x_t::Int, col::Int, alpha_0::Array{Float64,1})
        alpha = params()
        return utilsComputeSBF(x_t, alpha_0, alpha[:,col])
    end

    return UpdateRule(
        updateAll,
        init,
        update,
        params,
        computeTheta,
        computeSBF,
        "varSMiLe($m, $updateAll)"
    )
end


### particle filtering

# m: p_c / (1 - p_c), where p_c = probability of change
# N: number of particles
# Nthrs: threshold number for resampling
function particleFiltering(m, N, Nthrs, updateAll = false)

    # init state
    chi_0 = Array{Float64,2}(undef, 0, 0)
    chi_t = Array{Float64,3}(undef, N, 0, 0)
    w = ones(N) ./ N

    # set inital state
    function init(alpha_0::Array{Float64,2})
        # chi_t is N x 2 x 2^m, m = window length
        chi_0 = alphaToChi(alpha_0) |> copy
        chi_t = @pipe alphaToChi(alpha_0) |>
                      copy |>
                      fill(_, N) |>
                      arrayOfArrayToMatrix
    end

    assignAllCols = (chi::Array{Float64,3}, col::Int) -> chi_t = chi
    assignCol = (chi::Array{Float64,3}, col::Int) -> chi_t[:,:,col] = chi[:,:,col]
    assignChi = updateAll ? assignAllCols : assignCol

    function update(x_t::Int, col::Int)
        # compute surprises
        sbfs = map(
            chi -> computeSBFFromChi(x_t, chi_0[:,col], chi[:,col]),
            Slice(chi_t, 2, 3)
        )
        sbf = weightedHarmonicMean(sbfs, w)

        # compute gammas
        gammas = computeGamma.(sbfs, m)
        gamma = computeGamma(sbf, m)

        # update weights
        wB = sbf ./ sbfs .* w
        w = (1.0 - gamma) .* wB + gamma .* w

        # sample h ~ Bernoulli(gammas)
        h = rand.(Bernoulli.(gammas))

        # compute N_eff and resample if needed
        Neff = inv(sum(w .^ 2))
        if Neff < Nthrs
            # resample particles based on current weights
            newParticles = rand(Multinomial(N, w))

            # transform
            chi = @pipe mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(newParticles)) |>
                        map(p -> chi_t[p,:,:], _) |>
                        arrayOfArrayToMatrix;
            assignChi(chi, col);

            w = ones(N) ./ N
        end

        # update chi
        chi = map(
            chi -> (1.0 - h[i]) .* chi + gamma .* chi_0,
            Slices(chi_t, 2, 3)
        ) |> arrayOfArrayToMatrix;
        assignChi(chi, col);

        # update chi
        chi_t[:,x_t+1,col] .+= 1
    end

    # get state
    function params()
        alpha = map(
            chiToAlpha,
            Slices(chi_t, 2, 3)
        ) |> arrayOfArrayToMatrix;

        return alpha, w
    end

    # compute theta
    function computeTheta(col::Int)
        alpha, _ = params()

        return map(
            a -> utilsComputeTheta(a[:,col]),
            Slices(alpha, 2, 3)
        ) .* w |> sum
    end

    # compute surprise
    function computeSBF(x_t::Int, col::Int, alpha_0::Array{Float64,1})
        alpha, _ = params()
        alpha_t = map(a -> a[:,col], Slices(alpha, 2, 3))

        return utilsComputeSBF(x_t, alpha_0, alpha_t, w)
    end

    return UpdateRule(
        updateAll,
        init,
        update,
        params,
        computeTheta,
        computeSBF,
        "particleFiltering($m, $N, $Nthrs, $updateAll)"
    )
end
