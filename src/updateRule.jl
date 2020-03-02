# updateRule.jl
using SpecialFunctions, Distributions

mutable struct UpdateRule
    init # function that initializes up inital parameters
    update # function that updates the current parameters
end

### Utils

# compute the modulation factor gamma
function computeGamma(sgm, m)
    return m * sgm / (1 + m * sgm)
end

# multivariate Beta function
# alpha: shape should be (*,)
function Beta(alpha)
    return prod(gamma.(alpha)) / gamma(sum(alpha))
end

# dirichlet distribution
# theta: shape should be (*,)
# alpha: shape should be (*,)
function dirichletProb(theta, alpha)
    @assert(size(theta) == size(alpha))
    @assert(sum(theta) == 1.0)
    return 1.0 / Beta(alpha) * prod(theta .^ (alpha .- 1))
end

# create an array of zeros of given shape
# except for a one at the position pos
function oneAtPos(pos, shape)
    arr = zeros(shape)
    arr[pos...] = 1
    return arr
end

# compute "Generative Model surprise" for observation xt
# chi0: shape should be (2,) (binary signal)
# nu0: should be an integer
# chi: shape should be (2,) (binary signal)
# nu: should be an integer
function computeSGM(xt, chi0, nu0, chi, nu)
    # normalization function is 1 / Beta()
    function f(c, n)
        return 1.0 / Beta(c .+ 1)
    end

    # probability under given state
    function p(c, n)
        # we have to add 1 to xt since in Julia arrays start at 1
        return f(c, n) / f(c + oneAtPos((xt + 1), size(c)), n + 1)
    end

    # compute probability under state 0 and t
    p0 = p(chi0, nu0)
    pt = p(chi, nu)

    # surprise is ratio of probabilities
    return p0 / pt
end

# compute weighted harmonic mean
# arr should be of shape (*,)
function weightedHarmonicMean(arr, w)
    s = 0.0
    n = length(arr)
    @assert(length(w) == n)
    for i in 1 : n
        @inbounds s += w[i] * inv(a[i])
    end
    return sum(w) / s
end

### Perfect integration
function perfect()

    function update(t, transitions, alpha0)
        # update to Dirichlet prior is simply adding the occurences to alpha
        allTransitions = dropdims(sum(transitions, dims = 3), dims = 3)
        return alpha0 + allTransitions
    end

    return UpdateRule(() -> nothing, update)
end

### Leaky integration
function leaky(w)

    function update(t, transitions, alpha0)
        # first we weigh each transition by how far in the past it is
        decay = exp.(-1.0 / w * (t:-1:1))
        weightedTransitions = decay .* transitions

        # update to Dirichlet prior is simply
        # adding the weighted occurences to alpha
        allTransitions = dropdims(sum(weightedTransitions, dims = 3), dims = 3)
        return alpha0 + allTransitions
    end

    return UpdateRule(() -> nothing, update)
end

### Variational SMiLe
function varSMiLe(m)

    # init state
    chi = nothing
    nu = nothing

    function init(alpha0)
        # shape should be (2, (2, 2^m)) (m = window length)
        # shape[1] == 2: for chi_{(0)} or chi_{(t)}
        # shape[2][1] == 2: for value of x_t (binary signal)
        # shape[2][2] == 2^m: for every possible sequence of length m before x_t
        chi = fill(alpha0, 2)

        # shape should be (2, (2^m)) (m = window length)
        # shape[1] == 2: for nu_{(0)} or nu_{(t)}
        # shape[2][1] == 2^m: for every possible sequence of length m before x_t
        nu = fill(zeros(size(alpha0)[2]), 2)
    end

    # update rule
    function update(t, transitions, alpha0)
        # observe next x_t
        trans = argmax(transitions[:,:,end])
        xt = trans[1] - 1
        idx = trans[2]

        # compute surprise and modulation factor gamma
        sgm = computeSGM(xt, chi[1][:,idx], nu[1][idx],
                             chi[2][:,idx], nu[2][idx])
        gamma = computeGamma(sgm, m)

        # update chi and nu
        chi[2][:,idx] = (1 - gamma) * chi[2][:,idx] + gamma * chi[1][:,idx]
        chi[2][xt+1,idx] += 1

        nu[2][idx] = (1 - gamma) * nu[2][idx] + gamma * nu[1][idx] + 1

        # compute alpha to return
        return chi[2] .+ 1.0
    end

    return UpdateRule(init, update)
end

### Particle filtering
function particleFiltering(m,N,Nthrs)

    # m = p_c / (1 - p_c)

    # init state
    chi = nothing
    nu = nothing
    w = nothing

    function init(alpha0)
        # shape should be (2, N, (2, 2^m)) (m = window length)
        # shape[1] == 2: for chi_{(0)} or chi_{(t)}
        # shape[2] == N: for every particle i in 1:N
        # shape[3][1] == 2: for value of x_t (binary signal)
        # shape[3][2] == 2^m: for every possible sequence of length m before x_t
        chi = fill(alpha0, (2,N))

        # shape should be (2, N, (2^m)) (m = window length)
        # shape[1] == 2: for nu_{(0)} or nu_{(t)}
        # shape[2] == N: for every particle i in 1:N
        # shape[3][1] == 2^m: for every possible sequence of length m before x_t
        nu = fill(zeros(size(alpha0)[2]), (2,N))

        # shape should be (N,)
        # shape[1] == N: for every particle i in 1:N
        w = ones(N) ./ N
    end

    # alpha: shape should be (2,) (binary signal)
    function probXGivenAlpha(xt, alpha)
        shape = size(alpha)
        n = shape[1]

        # calc thetas
        thetas = (alpha - 1.0) ./ (sum(alpha) - n)

        # calc probability
        return dirichletProb(thetas, alpha .+ 1 + oneAtPos((xt + 1), shape))
    end

    # update rule
    function update(t, transitions, alpha0)
        # observe x_{t+1}
        trans = argmax(transitions[:,:,end])
        xt = trans[1] - 1
        idx = trans[2]

        # compute surprises and gamma
        sgms = computeSGM.(xt, chi[1,:][:,idx], nu[1,:][idx],
                               chi[2,:][:,idx], nu[2,:][idx])
        sgm = weightedHarmonicMean(sgms, w)
        gamma = computeGamma(sgm, m)

        # compute Bayesian update weights
        probXByParticle = probXGivenAlpha.(xt, chi[2,:][:,idx] .+ 1)
        wB = probXByParticle ./ sum(w .* probXByParticle)

        # update weights
        w = (1 - gamma) .* wB + gamma .* w

        # sample h ~ Bernoulli(gamma)
        bernoulli = Bernoulli(gamma)
        sample = () -> rand(bernoulli, N)
        h = sample()

        # compute N_eff
        Neff = inv(sum(w .^ 2))
        if Neff < Nthrs:
            # resample h
            h = sample()

        # update chi and nu
        # if h[i] == 0, use (t), else use (0)
        chi[2,:][:,idx] = (1 .- h) .* chi[2,:][:,idx] .+ h .* chi[1,:][:,idx]
        chi[2,:][xt+1,idx] .+= 1

        nu[2,:][idx] = (1 .- h) .* nu[2,:][idx] .+ h .* nu[1,:][idx] .+ 1

        # compute alpha to return
        # todo: return new distribution?
        # can't return alpha values since we have a
        # weighted sum of Dirichlet distributions
        return alpha0
    end

    return UpdateRule(init, update)
end
