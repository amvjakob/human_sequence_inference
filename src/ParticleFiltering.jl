# ParticleFiltering.jl

using Distributions

include("UpdateRule.jl")
include("utils.jl")


"""
    ParticleFiltering(m, nparticles, nthreshold, prior; N = 2, updateallcols = true) -> UpdateRule

Create a new particle filtering learning rule, with change factor `m`
(where ``m = \\frac{p_c}{1 - p_c}`` and ``p_c`` is the probability of
change), `nparticles` particles, sampling threshold number `nthreshold`
and prior over different window lengths `prior`. 

Optionally, `N` represents the number of different elements in the 
signal to be decoded and `updateallcols` represents whether to leak
the entire state or just the one  corresponding to the current observation.
"""
function ParticleFiltering(m::Float64,
                           nparticles::Integer,
                           nthreshold::Integer,
                           prior::Array{Float64,1};
                           N = 2,
                           updateallcols = true)

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

    # compute neff and resample if needed
    neff = inv(sum(w .^ 2))
    if neff < nthreshold
      # resample particles based on current weights
      # this returns an array a of length nparticles, 
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
