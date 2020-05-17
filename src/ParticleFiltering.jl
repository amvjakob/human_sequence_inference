# ParticleFiltering.jl

using Distributions, Pipe, JuliennedArrays

include("utils.jl")
include("UpdateRule.jl")

### particle filtering

# m: p_c / (1 - p_c), where p_c = probability of change
# N: number of particles
# Nthrs: threshold number for resampling
# updateall: whether to leak all cols or just the current one
function ParticleFiltering(m, N, Nthrs, updateall = false)

  # set initial state
  chi_0 = Array{Float64,2}(undef, 0, 0)
  chi_t = Array{Float64,3}(undef, N, 0, 0)
  w = ones(N) ./ N

  # init state state
  function init(alpha_0::Array{Float64,2})
    # chi_t is N x 2 x 2^m, m = window length
    chi_0 = copy(alpha_0 .- 1)
    chi_t = fill(copy(alpha_0 .- 1), N) |> unwrap
  end

  # update state
  function update(x_t::Integer, col::Integer)
    # compute surprises
    sbfs = map(
      c -> compute_sbf(x_t, chi_0[:,col] .+ 1, c[:,col] .+ 1),
      Slices(chi_t, 2, 3)
    )
    sbf = weighted_harmonic_mean(sbfs, w)

    # compute gammas
    gammas = compute_gamma.(sbfs, m)
    gamma  = compute_gamma(sbf, m)

    # update weights
    wB = sbf ./ sbfs .* w
    w = (1.0 - gamma) .* wB + gamma .* w

    # sample h ~ Bernoulli(gammas)
    h = rand.(Bernoulli.(gammas))

    # compute N_eff and resample if needed
    Neff = inv(sum(w .^ 2))
    if Neff < Nthrs
      # resample particles based on current weights
      # this returns a vector v of length N, 
      # where v[i] represents the number of old particles i we keep
      particles = rand(Multinomial(N, w))

      # transform particles to chi values
      chi = @pipe mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(particles)) |>
                  map(p -> chi_t[p,:,:], _) |>
                  unwrap;

      # assign chi
      if updateall
        chi_t = chi
      else
        chi_t[:,:,col] = chi[:,:,col]
      end

      # reset weights to uniform
      w = ones(N) ./ N
    end

    # "surprise modulation"
    chi = map(
      i -> (1 - h[i]) .* chi_t[i,:,:] + h[i] .* chi_0,
      eachindex(h)
    ) |> unwrap;
    
    # assign chi
    if updateall
      chi_t = chi
    else
      chi_t[:,:,col] = chi[:,:,col]
    end

    # update chi_t
    chi_t[:, x_t + 1, col] .+= 1

    # return surprise-modulated learning rate
    return gamma
  end

  # get state
  function params()
    return chi_t .+ 1, w
  end

  # compute theta
  function gettheta(col::Integer, x = 1)::Float64
    alpha = chi_t .+ 1

    return map(
        a -> compute_theta(a[:,col], x),
        Slices(alpha, 2, 3)
    ) .* w |> sum
  end

  # compute surprise
  function getsbf(x_t::Integer, col::Integer)
    alpha = chi_t .+ 1
    alpha_t = map(a -> a[:,col], Slices(alpha, 2, 3))

    return compute_sbf(x_t, chi_0[:,col] .+ 1, alpha_t, w)
  end

  return UpdateRule(
    updateall,
    init,
    update,
    params,
    gettheta,
    getsbf,
    "ParticleFiltering($m, $N, $Nthrs, $updateall)"
  )
end

### particle filtering update rule with inference over m

# m: p_c / (1 - p_c), where p_c = probability of change
# N: number of particles
# Nthrs: threshold number for resampling
# updateall: whether to leak all cols or just the current one
function ParticleFilteringInference(m, Nparticles, Nthrs, updateall = false)
  # set initial state
  models   = Array{UpdateRule,1}(undef, 0)
  priors_0 = Array{Float64,1}(undef, 0)
  priors_t = Array{Float64,1}(undef, 0)

  # init state
  function init(prior::Array{Float64,1}, N = 2)
    models   = Array{UpdateRule,1}(undef, length(prior))
    priors_0 = copy(prior)
    priors_t = copy(prior)

    for i in 1:length(prior)
      len = i - 1

      # use uniform Dirichlet prior for each submodel
      model = ParticleFiltering(len, Nparticles, Nthrs, updateall)
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
    thetas = getthetas(cols, x_t) # this is an array of p(x_t | m) for all m
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

    # compute p_integrate and p_reset
    p_int = priors_t .* thetas 
    p_res = priors_0 .* thetas

    # normalize p_integrate and p_reset
    p_int = p_int / sum(p_int)
    p_res = p_res / sum(p_res)

    # update prior    
    priors_t = (1 .- gammas) .* p_int .+ gammas .* p_res

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
      i -> models[i].getsbf(x_t,max(cols[i],1)),
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
    "ParticleFilteringInference($m, $Nparticles, $Nthrs, $updateall)"
  )
end

