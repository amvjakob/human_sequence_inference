# updateRule.jl
using Distributions

include("utils.jl")


### update rule skeleton

mutable struct UpdateRule
    # flag that indicates whether to update all parameters
    # or just the current column
    updateAll::Bool

    # function that initializes up inital parameters
    init

     # function that updates the current parameters
    update

    # function that returns the current parameters
    params
end


### perfect integration

function perfect()

    # set inital state
    alpha = nothing

    # init state
    function init(alpha_0)
        alpha = deepcopy(alpha_0)
    end

    # update state
    function update(x_t, col, x_t_weight = 1)
        # update is simply adding to the corresponding alpha
        alpha[x_t+1, col] += x_t_weight
    end

    # get state
    function params()
        return alpha
    end

    return UpdateRule(
        false,
        init,
        update,
        params
    )
end


### leaky integration

# w: leak factor
function leaky(w, updateAll = false)

    # set inital state
    alpha = nothing
    decay = exp(-1.0 / w)

    # init state
    function init(alpha_0)
        alpha = deepcopy(alpha_0)
    end

    # update state
    function updateCol(x_t, col, x_t_weight = 1)
        alpha[:,col] = decay * (alpha[:,col] .- 1) .+ 1
        alpha[x_t+1,col] += x_t_weight
    end

    function updateAllCols(x_t, col, x_t_weight = 1)
        alpha[:,:] = decay * (alpha[:,:] .- 1) .+ 1
        alpha[x_t+1,col] += x_t_weight
    end

    # get state
    function params()
        return alpha
    end

    return UpdateRule(
        updateAll,
        init,
        updateAll ? updateAllCols : updateCol,
        params
    )
end


### variational SMiLe

# m: p_c / (1 - p_c), where p_c = probability of change
function varSMiLe(m, updateAll = false)

    # set initial state
    chi_0 = nothing
    chi_t = nothing

    # init state
    function init(alpha_0)
        # shape of chi is (2, 2^m) (m = window length)
        # shape[1] == 2: for value of x_t (binary signal)
        # shape[2] == 2^m: for every possible sequence of length m before x_t
        chi_0 = alphaToChi(alpha_0)
        chi_t = alphaToChi(alpha_0)
    end

    # update state
    function update(x_t, col, x_t_weight = 1.0)
        # compute surprise
        sbf = computeSBFFromChi(x_t, chi_0[:,col], chi_t[:,col])
        gamma = computeGamma(sbf, m)

        # update chi
        if updateAll
            chi_t = (1.0 - gamma) .* chi_t + gamma .* chi_0
        else
            chi_t[:,col] = (1.0 - gamma) .* chi_t[:,col] + gamma .* chi_0[:,col]
        end

        # add observation
        chi_t[x_t+1,col] += x_t_weight
    end

    # get state
    function params()
        # return corresponding alpha
        return chiToAlpha(chi_t)
    end

    return UpdateRule(
        updateAll,
        init,
        update,
        params
    )
end


### particle filtering

# m: p_c / (1 - p_c), where p_c = probability of change
# N: number of particles
# Nthrs: threshold number for resampling
function particleFiltering(m, N, Nthrs, updateAll = false)

    # init state
    chi_0 = nothing
    chi_t = nothing
    w = nothing

    # set inital state
    function init(alpha_0)
        # shape should be (N, 2, 2^m) (m = window length)
        # shape[1] == N: for every particle i in 1:N
        # shape[2] == 2: for value of x_t (binary signal)
        # shape[3] == 2^m: for every possible sequence of length m before x_t
        chi_0 = zeros((N, size(alpha_0)...))
        chi_t = zeros((N, size(alpha_0)...))

        for i in 1:N
            chi_0[i,:,:] = alphaToChi(alpha_0)
            chi_t[i,:,:] = alphaToChi(alpha_0)
        end

        # shape should be (N,)
        # shape[1] == N: for every particle i in 1:N
        w = ones(N) ./ N
    end

    # update rule
    function update(x_t, col, x_t_weight = 1)
        # compute surprises
        sbfs = zeros(N)
        for i in 1:N
            sbfs[i] = computeSBFFromChi(x_t, chi_0[i,:,col], chi_t[i,:,col])
        end
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
            chi = zeros(size(chi_t))
            counter = 0

            # copy new particles
            for (i, nParticles) in enumerate(newParticles)
                for j in 1:nParticles
                    if updateAll
                        # resample all columns
                        chi[j+counter,:,:] = chi_t[i,:,:]
                    else
                        # only resample given column
                        chi[j+counter,:,col] = chi_t[i,:,col]
                    end
                end
            end

            # set new particles and weights
            chi_t = deepcopy(chi)
            w = ones(N) ./ N
        end

        # update chi
        for i in 1:N
            # if h[i] == 0, use (t), else use (0)
            if updateAll
                chi_t[i,:,:] = (1.0 - h[i]) .* chi_t[i,:,:] + gamma .* chi_0[i,:,:]
            else
                chi_t[i,:,col] = (1.0 - h[i]) .* chi_t[i,:,col] + gamma .* chi_0[i,:,col]
            end

            chi_t[i,x_t+1,col] += x_t_weight
        end
    end

    # get state
    function params()
        return chi_t, w
    end

    return UpdateRule(
        updateAll,
        init,
        update,
        params
    )
end
