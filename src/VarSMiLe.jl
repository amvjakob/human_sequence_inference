# VarSMiLe.jl

include("UpdateRule.jl")
include("utils.jl")


"""
    VarSMiLe(m, prior; N = 2, updateallcols = false) -> UpdateRule

Create a new variational SMiLe (Surprise Minimization Learning) learning
rule, with change factor `m` (where ``m = \\frac{p_c}{1 - p_c}`` and
``p_c`` is the probability of change) and prior over different window 
lengths `prior`. 

Optionally, `N` represents the number of different elements in the 
signal to be decoded and `updateallcols` represents whether to leak
the entire state or just the one corresponding to the current observation.
"""
function VarSMiLe(m::Float64,
                  prior::Array{Float64,1};
                  N = 2,
                  updateallcols = false)

  # check argument validity
  @assert(m >= 0)
  @assert(sum(prior) === 1.0)
  @assert(N >= 2)

  # inital state of prior is simply a copy of the provided argument
  prior_0::Array{Float64,1} = copy(prior)
  prior_t::Array{Float64,1} = copy(prior)
  
  # chi is alpha - 1
  chi_0::Array{Array{Float64,2},1} = map(i -> zeros(N, N^(i-1)), eachindex(prior))
  chi_t::Array{Array{Float64,2},1} = deepcopy(chi_0)

  # whether the prior is fixed (1 for one model and 0 for the rest)
  isfixed::Bool = findmax(prior)[1] === 1.0


  # reset state
  function reset()

    # reset prior
    prior_t = copy(prior_0)

    # reset chi
    chi_t = deepcopy(chi_0)

  end

  
  # compute thetas
  function getthetas(x::Integer, cols::Array{<:Integer,1})::Array{Float64,1}

    return map(chi_t, cols) do chi, col
      # if col is 0 (partial window), we set it to 1,
      # since this col will still be unmodified
      c = max(col, 1)
      # compute theta from chi
      return compute_theta(x, chi[:,c] .+ 1)
    end

  end


  # compute theta
  function gettheta(x::Integer, cols::Array{<:Integer,1})::Float64
    return sum(prior_t .* getthetas(x, cols))
  end


  # compute surprise
  function getsbf(x::Integer, cols::Array{<:Integer,1})::Float64
    return 1.0 / N / gettheta(x, cols)
  end


  # get params
  function getposterior()::Array{Float64,1}
    return prior_t
  end


  # update state
  function update(x_t::Integer, cols::Array{<:Integer,1})

    # get params before update
    chi_t_before = deepcopy(chi_t)
    
    # update models
    gammas = ones(length(chi_t))
    for i in eachindex(chi_t)
      col = cols[i]
      if col > 0
        # compute gamma
        sbf = compute_sbf(x_t, chi_0[i][:,col] .+ 1, chi_t[i][:,col] .+ 1)
        gamma = compute_gamma(sbf, m)
        gammas[i] = gamma

        # surprise modulation
        if updateallcols
          chi_t[i]        = (1.0 - gamma) .* chi_t[i]        + gamma .* chi_0[i]
        else
          chi_t[i][:,col] = (1.0 - gamma) .* chi_t[i][:,col] + gamma .* chi_0[i][:,col]
        end

        # update chi_t
        chi_t[i][x_t + 1, col] += 1
      else
        gamma = compute_gamma(1.0, m)
        gammas[i] = gamma
      end
    end

    # update posterior
    if !isfixed
      for i in eachindex(chi_t)
        # compute factor
        lnprob = 0.0
        for col in 1:size(chi_t[i], 2)
          lnprob += ln_beta_fn(chi_t[i][:,col] .+ 1) - (
                      (1 - gammas[i]) * ln_beta_fn(chi_t_before[i][:,col] .+ 1) + 
                      gammas[i] * ln_beta_fn(ones(N))
                    )
        end
        
        prior_t[i] = exp(lnprob) * prior_t[i] ^ (1 - gammas[i]) * prior_0[i] ^ gammas[i]
      end

      # normalize prior
      prior_t ./= sum(prior_t)
    end

  end


  return UpdateRule(
    reset,
    gettheta,
    getsbf,
    getposterior,
    update,
    updateallcols,
    "VarSMiLe($m, $prior, $updateallcols)"
  )
end
