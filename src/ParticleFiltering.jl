# ParticleFiltering.jl

using Distributions, JuliennedArrays

include("utils.jl")
include("UpdateRule.jl")

### particle filtering

# m: p_c / (1 - p_c), where p_c = probability of change
# N: number of particles
# Nthrs: threshold number for resampling
# updateall: whether to leak all cols or just the current one
"""function ParticleFiltering(m, N, Nthrs, updateall = true)

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
      # this returns an array a of length N, 
      # where a[i] represents the number of old particles i we keep
      particles = rand(Multinomial(N, w))
      # map to array a, where a[i] is the index of the particle we keep
      particles = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(particles))

      # transform particles to chi values
      chi = map(p -> chi_t[p,:,:], particles) |> unwrap

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
      i -> (1 - h[i]) .* chi_t[i,:,:] .+ h[i] .* chi_0,
      eachindex(h)
    ) |> unwrap
    
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
    "ParticleFiltering(m, N, Nthrs, updateall)"
  )
end"""

### particle filtering update rule

# m: p_c / (1 - p_c), where p_c = probability of change
# nparticles: number of particles
# nthreshold: threshold number for resampling
# prior: prior over window length
# N: number of different elements in signal (binary = 2)
# updateallcols: whether to update all cols or just the current one

function ParticleFiltering(m::Float64, nparticles::Integer,
  nthreshold::Integer, prior::Array{Float64,1}; 
  N = 2, updateallcols = true)

  # check argument validity
  @assert(m >= 0)
  @assert(sum(prior) === 1.0)
  @assert(N >= 2)

  # set initial state
  prior_0::Array{Float64,1} = copy(prior)
  prior_t::Array{Array{Float64,1},1} = fill(copy(prior), nparticles)

  # chi = alpha - 1
  chi_0::Array{Array{Float64,2},1} = map(i -> zeros(N, N^(i-1)), eachindex(prior))
  chi_t::Array{Array{Array{Float64,2},1},1} = fill(chi_0, nparticles) |> deepcopy

  # particle weights
  w = ones(nparticles) ./ nparticles

  # whether the prior is fixed (1 for one model and 0 for the rest)
  isfixed::Bool = findmax(prior)[1] === 1.0


  # reset state
  function reset()

    # reset prior
    prior_t = fill(prior_0, nparticles) |> deepcopy

    # reset chi
    chi_t = fill(chi_0, nparticles) |> deepcopy

    # reset particle weights
    w = ones(nparticles) ./ nparticles

  end


  # get thetas (plural) for a given particle
  function getparticlethetas(particle::Integer, x::Integer, cols::Array{<:Integer,1})::Array{Float64}

    return map(chi_t[particle], cols) do chi, col
      # if col is 0 (partial window), we set it to 1,
      # since this col will still be unmodified
      c = max(col, 1)
      # compute theta from chi
      return compute_theta(x, chi[:,c] .+ 1)
    end

  end


  # get theta for a given particle
  function getparticletheta(particle::Integer, x::Integer, cols::Array{<:Integer,1})::Float64
    return sum(prior_t[particle] .* getparticlethetas(particle, x, cols))
  end


  # get Bayes Factor surprise for a given particle
  function getparticlesbf(particle::Integer, x::Integer, cols::Array{<:Integer,1})::Float64
    # sbf is theta_0 / theta_t
    # theta_0 is 1 / N
    return 1.0 / N / getparticletheta(particle, x, cols)
  end


  # get theta 
  function gettheta(x::Integer, cols::Array{<:Integer,1})::Float64
    return sum(w .* getparticletheta.(1:nparticles, x, Ref(cols)))
  end


  # get Bayes Factor surprise
  function getsbf(x::Integer, cols::Array{<:Integer,1})::Float64
    return 1.0 / N / gettheta(x, cols)
  end


  # get params
  function getposterior()::Array{Float64,1}
    return sum(w .* prior_t)
  end


  # update state
  function update(x_t::Integer, cols::Array{<:Integer,1})

    # update posterior
    if !isfixed
      for i in eachindex(prior_t)
        prior_t[i] = prior_t[i] .* getparticlethetas(i, x_t, cols)
        prior_t[i] = prior_t[i] ./ sum(prior_t[i])
      end
    end

    # compute surprises
    sbfs = getparticlesbf.(1:nparticles, x_t, Ref(cols))
    sbf  = weighted_harmonic_mean(sbfs, w)

    # compute gammas
    gammas = compute_gamma.(sbfs, m)
    gamma  = compute_gamma(sbf, m)

    # update weights
    wB = sbf ./ sbfs .* w
    w = (1.0 - gamma) .* wB + gamma .* w

    # sample h ~ Bernoulli(gammas)
    h = rand.(Bernoulli.(gammas))

    # compute n_eff and resample if needed
    neff = inv(sum(w .^ 2))
    if neff < nthreshold
      # resample particles based on current weights
      # this returns an array a of length Nparticles, 
      # where a[i] represents the number of old particles i we keep
      particles = rand(Multinomial(nparticles, w))
      # map to array a, where a[i] is the index of the particle we keep
      particles = mapreduce(x -> fill(x[1], x[2]), vcat, enumerate(particles))

      # transform particles to chi values
      chi = map(p -> chi_t[p], particles)

      # transform particles to posterior values
      prior_t = map(p -> prior_t[p], particles) |> deepcopy

      # assign chi
      if updateallcols
        chi_t = deepcopy(chi)
      else
        chi_t .= map(chi_t, chi) do chi_particle_old, chi_particle_new
          return map(chi_particle_old, chi_particle_new, cols) do chi_old, chi_new, col
            # only update current column
            col = max(col, 1)
            chi_old[:,col] = chi_new[:,col]
            return chi_old
          end
        end
      end

      # reset weights to uniform
      w = ones(nparticles) ./ nparticles
    end

    # surprise modulation
    chi = map((hval, chi) -> (1 - hval) .* chi .+ hval .* chi_0, h, chi_t)

    # assign chi
    if updateallcols
      chi_t = deepcopy(chi)
    else
      for i in eachindex(chi_t)
        for j in eachindex(chi_t[i])
          col = max(cols[j], 1)
          chi_t[i][j][:,col] .= chi[i][j][:,col]
        end
      end
    end

    # update chi_t
    for j in eachindex(cols)
      if cols[j] > 0
        for i in eachindex(chi_t)
          chi_t[i][j][x_t + 1, cols[j]] += 1
        end          
      end
    end

  end


  return UpdateRule(
    reset,
    gettheta,
    getsbf,
    getposterior,
    update,
    updateallcols,
    "PartFiltering($m, $nparticles, $nthreshold, $prior, $updateallcols)"
  )
end
