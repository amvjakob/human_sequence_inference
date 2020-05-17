# Leaky.jl

include("utils.jl")
include("UpdateRule.jl")

### leaky integration

# utility functions
function build_rules_leaky(ws)
  return map(w -> Leaky(w, true), ws)
end


# w: leak factor
# updateall: whether to leak all cols or just the current one
function Leaky(w, updateall = false)

  # perfect integration is a special case of leaky integration
  if w == Inf
    return Perfect()
  end

  # set inital state
  # chi is alpha - 1
  chi_0 = Array{Float64,2}(undef, 0, 0)
  chi_t = Array{Float64,2}(undef, 0, 0)

  # decay factor
  η = exp(-1.0 / w)

  # init state
  function init(alpha_0::Array{Float64,2})
    chi_0 = copy(alpha_0 .- 1)
    chi_t = copy(alpha_0 .- 1)
  end

  # update state
  function update(x_t::Integer, col::Integer)
    # update chi_t
    chi_t[x_t + 1, col] += 1

    # leak memory
    if updateall
      chi_t        = η * chi_t
    else
      chi_t[:,col] = η * chi_t[:,col]
    end
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
  function getsbf(x_t::Integer, col::Integer)::Float64
    return compute_sbf(x_t, chi_0[:,col] .+ 1, chi_t[:,col] .+ 1)
  end

  return UpdateRule(
    updateall,
    init,
    update,
    params,
    gettheta,
    getsbf,
    "Leaky($w, $updateall)"
  )
end

### perfect integration
# perfect integration is a special case of leaky integration with w = ∞

function Perfect()

  # set inital state
  alpha_t = Array{Float64,2}(undef, 0, 0)
  alpha_0 = Array{Float64,2}(undef, 0, 0)

  # init state
  function init(alpha_0_arg::Array{Float64,2})
    alpha_t = copy(alpha_0_arg)
    alpha_0 = copy(alpha_0_arg)
  end

  # update state
  function update(x_t::Integer, col::Integer)
    alpha_t[x_t + 1,col] += 1
  end

  # get state
  function params()
    return alpha_t
  end

  # compute theta
  function gettheta(col::Integer, x = 1)::Float64
    return compute_theta(alpha_t[:,col], x)
  end

  # compute surprise
  function getsbf(x_t::Integer, col::Integer)::Float64
    return compute_sbf(x_t, alpha_0[:,col], alpha_t[:,col])
  end

  return UpdateRule(
    false,
    init,
    update,
    params,
    gettheta,
    getsbf,
    "Perfect()"
  )
end


### leaky update rule with inference over m

# w: leak factor
# full_leaky: whether to leak prior over m or just thetas
# updateall: whether to leak all cols or just the current one
function LeakyInference(w, full_leaky = false, updateall = false)
  # set initial state
  models = Array{UpdateRule,1}(undef, 0)
  priors = Array{Float64,1}(undef, 0)

  # decay factor
  η = w == Inf ? 1 : exp(-1.0 / w)

  # init state
  function init(prior::Array{Float64,1}, N = 2)
    models = Array{UpdateRule,1}(undef, length(prior))
    priors = copy(prior)

    for i in 1:length(prior)
      m = i - 1

      # use uniform Dirichlet prior for each submodel
      model = w == Inf ? Perfect() : Leaky(w, updateall)
      model.init(ones(N, N^m))

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
    # update prior
    thetas = getthetas(cols, x_t) # this is an array of p(x_t | m) for all m
    priors = priors .* thetas

    # leak prior
    if full_leaky
      priors = priors .^ η 
    end

    # normalize prior
    priors = priors / sum(priors)

    # update models
    for i in eachindex(models)
      # update model if col > 0 (ignore partial window)
      cols[i] > 0 && models[i].update(x_t, cols[i])
    end
  end

  # get params
  function params()
    return map(m -> m.params(), models), priors
  end

  # compute theta
  function gettheta(cols::Array{<:Integer,1}, x = 1)::Float64
    return sum(priors .* getthetas(cols, x))
  end

  # compute surprise
  function getsbfs(x_t::Integer, cols::Array{<:Integer,1})::Array{Float64,1}
    return map(
      i -> models[i].getsbf(x_t, max(cols[i],1)),
      eachindex(models)
    )
  end

  function getsbf(x_t::Integer, cols::Array{<:Integer,1})::Float64
    return sum(priors .* getsbfs(x_t, cols))
  end

  return UpdateRule(
    updateall,
    init,
    update,
    params,
    gettheta,
    getsbf,
    w == Inf ? "PerfectInference" : "LeakyInference($w, $updateall)"
  )
end
