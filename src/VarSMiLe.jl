# VarSMiLe.jl

include("utils.jl")
include("UpdateRule.jl")

### variational SMiLe

# utility functions
function build_rules_varsmile(ms)
  return map(m -> VarSMiLe(m, true), ms)
end


# m: p_c / (1 - p_c), where p_c = probability of change
# updateall: whether to leak all cols or just the current one
function VarSMiLe(m, updateall = false)

  # set initial state
  # chi is alpha - 1
  chi_0 = Array{Float64,2}(undef, 0, 0)
  chi_t = Array{Float64,2}(undef, 0, 0)

  # init state
  function init(alpha_0::Array{Float64,2})
    chi_0 = copy(alpha_0 .- 1)
    chi_t = copy(alpha_0 .- 1)
  end

  # update state
  function update(x_t::Integer, col::Integer)
    # compute gamma
    sbf = compute_sbf(x_t, chi_0[:,col] .+ 1, chi_t[:,col] .+ 1)
    gamma = compute_gamma(sbf, m)

    # surprise modulation
    if updateall
      chi_t        = (1.0 - gamma) .* chi_t        + gamma .* chi_0
    else
      chi_t[:,col] = (1.0 - gamma) .* chi_t[:,col] + gamma .* chi_0[:,col]
    end

    # update chi_t
    chi_t[x_t + 1, col] += 1

    # return surprise-modulated learning rate
    return gamma
  end

  # get state
  function params()
    return chi_t .+ 1
  end

  # compute theta
  function gettheta(col::Integer, x = 1)::Float64
    return compute_theta(chi_t[:,col] .+ 1, x)
  end

  # compute surprise
  function getsbf(x_t::Integer, col::Integer)
    return compute_sbf(x_t, chi_0[:,col] .+ 1, chi_t[:,col] .+ 1)
  end

  return UpdateRule(
    updateall,
    init,
    update,
    params,
    gettheta,
    getsbf,
    "VarSMiLe($m, $updateall)"
  )
end


### variational SMiLe update rule with inference over m

# m: p_c / (1 - p_c), where p_c = probability of change
# updateall: whether to leak all cols or just the current one
function VarSMiLeInference(m, updateall = false)
  # set initial state
  models   = Array{UpdateRule,1}(undef, 0)
  priors_0 = Array{Float64,1}(undef, 0)
  priors_t = Array{Float64,1}(undef, 0)

  # init state
  function init(prior::Array{Float64,1}, N = 2)
    models = Array{UpdateRule,1}(undef, length(prior))
    priors_0 = copy(prior)
    priors_t = copy(prior)

    for i in 1:length(prior)
      len = i - 1

      # use uniform Dirichlet prior for each submodel
      model = VarSMiLe(m, updateall)
      model.init(ones(N, N^len))

      models[i] = model
    end
  end

  # get thetas
  function getthetas(cols::Array{<:Integer,1}, x = 1)::Array{Float64,1}
    thetas = zeros(length(cols))
    for i in eachindex(thetas)
      # if col is 0 (partial window), we set it to 1 (since this col will still be unîform)
      thetas[i] = models[i].gettheta(max(cols[i], 1), x)
    end

    return thetas
  end

  # update state
  function update(x_t::Integer, cols::Array{<:Integer,1})
    # get params before update
    params_before = map(m -> m.params(), models)
    
    # update models
    # gamma = 1 is equivalent to "forgetting" everything
    gammas = ones(length(models))
    for i in eachindex(models)
      # update model if col > 0 (ignore partial window)
      if cols[i] > 0
        gammas[i] = models[i].update(x_t, cols[i])
      end
    end

    # get params after update
    params_after = map(m -> m.params(), models)

    # update prior
    N = size(params_before[1], 1)
    for i in eachindex(priors_t)
      len = i - 1

      # compute factor
      lnprob = 0.0
      for col in 1:N^len
        lnprob += ln_beta_fn(params_after[i][:,col]) - ((1 - gammas[i]) * ln_beta_fn(params_before[i][:,col]) + gammas[i] * beta_fn(ones(N)))
      end
      
      if priors_t[i] > 0 && priors_0[i] > 0
        priors_t[i] = exp(lnprob + (1 - gammas[i]) * log(priors_t[i]) + gammas[i] * log(priors_0[i]))
      end
    end

    # normalize prior
    priors_t = priors_t / sum(priors_t)
  end

  # get params
  function params()
    return map(m -> m.params(), models), priors_t
  end

  # compute theta
  function gettheta(cols::Array{<:Integer,1}, x = 1)::Float64
    return sum(priors_t .* getthetas(cols, x))
  end

  # compute surprise
  function getsbfs(x_t::Integer, cols::Array{<:Integer,1})::Array{Float64,1}
    return map(
      i -> models[i].getsbf(x_t, max(cols[i],1)),
      eachindex(models)
    )
  end

  function getsbf(x_t::Integer, cols::Array{<:Integer,1})::Float64
    return sum(priors_t .* getsbfs(x_t, cols))
  end

  return UpdateRule(
    updateall,
    init,
    update,
    params,
    gettheta,
    getsbf,
    "VarSMiLeInference($m, $updateall)"
  )
end
