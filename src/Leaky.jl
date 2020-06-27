# Leaky.jl

include("UpdateRule.jl")
include("utils.jl")


"""
    Leaky(w, prior; N = 2, leakprior = false, updateallcols = false) -> UpdateRule

Create a new leaky integration learning rule, with leak factor `w` 
(where `w = Inf` is perfect integration) and prior over different window 
lengths `prior`. 

Optionally, `N` represents the number of different elements in the 
signal to be decoded, `leakprior` represents whether to also "leak"
the prior and `updateallcols` represents whether to leak the entire
state or just the one corresponding to the current observation.
"""
function Leaky(w::Union{Integer,Float64},
               prior::Array{Float64,1}; 
               N = 2,
               leakprior = false,
               updateallcols = false)

  # check argument validity
  @assert(w > 0)
  @assert(sum(prior) === 1.0)
  @assert(N >= 2)

  # inital state of prior is simply a copy of the provided argument
  prior_t::Array{Float64,1} = copy(prior)
  
  # chi is alpha - 1
  chi_0::Array{Array{Float64,2},1} = map(i -> zeros(N, N^(i-1)), eachindex(prior)) 
  chi_t::Array{Array{Float64,2},1} = deepcopy(chi_0)

  # decay factor
  η::Float64 = exp(-1.0 / w)

  # whether the prior is fixed (1 for one model and 0 for the rest)
  isfixed::Bool = findmax(prior)[1] === 1.0

  
  # reset state
  function reset()

    # reset prior
    prior_t = copy(prior)
  
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

    # update posterior
    if !isfixed
      prior_t .*= getthetas(x_t, cols) # this is an array of p(x_t | m) for all m

      if leakprior
        prior_t .^= η 
      end

      # normalize prior
      prior_t ./= sum(prior_t)
    end

    # update models
    chi_t .= map(chi_t, cols) do chi, col

      if col > 0
        # increment event count
        chi[x_t + 1, col] += 1

        # leak
        if updateallcols
          chi        .*= η
        else
          chi[:,col] .*= η
        end
      end

      return chi
    end
  end

  
  return UpdateRule(
    reset,
    gettheta,
    getsbf,
    getposterior,
    update,
    updateallcols,
    "Leaky($w, $prior, $leakprior, $updateallcols)"
  )
end
