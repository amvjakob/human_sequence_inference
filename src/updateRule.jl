# UpdateRule.jl

using LaTeXStrings

include("utils.jl")

### callback struct

# A Callback object is basically a wrapper around a function
# F: type of wrapped function
# R: return type of wrapped function
mutable struct Callback{F,R}
  # wrapped function
  run::F

  # constructor
  function Callback(fun::F, R) where F
    return new{F,R}(fun)
  end
end


### update rule

# An UpdateRule object represents a learning rule
mutable struct UpdateRule{I,U,P,T,S}
  # flag that indicates whether to update all parameters
  # or just the current column
  updateall::Bool

  # function that initializes inital parameters
  init::I

  # function that updates the current parameters
  update::U

  # function that returns the current parameters
  params::P

  # function that computes theta
  gettheta::T

  # function that computes the Surprise Bayes Factor
  getsbf::S

  # name of the update rule (useful for graphs)
  str
end


### utility functions


function build_models(rules::Array{UpdateRule,1},
                      ms::Array{Int,1},
                      names::Array{Str,1}) where {Str <: AbstractString}
  return map(
    m -> map(
      r -> Dict(
        "m" => m[1],
        "alpha_0" => ones(2,2^m[1]),
        "rule" => r,
        "name" => m[2]
      ),
      rules), 
    zip(ms, names)
  )
end

function build_models(rules::Array{UpdateRule{I,U,P,T,S},1},
                      ms::Array{Int,1},
                      names::Array{Str,1}) where {I,U,P,T,S,Str <: AbstractString}
  return map(
    m -> map(
      r -> Dict(
        "m" => m[1],
        "alpha_0" => ones(2,2^m[1]),
        "rule" => r,
        "name" => m[2]
      ),
      rules), 
    zip(ms, names)
  )
end

function build_models(rules::Array{UpdateRule,1}, ms::Array{Int,1})
  return build_models(rules, ms, map(m -> latexstring("m = $m"), ms))
end

function build_models(rules::Array{UpdateRule{I,U,P,T,S},1}, ms::Array{Int,1}) where {I,U,P,T,S}
  return build_models(rules, ms, map(m -> latexstring("m = $m"), ms))
end




