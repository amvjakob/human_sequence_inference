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
mutable struct UpdateRule{R,T,S,P,U}
    # function that initializes inital parameters
  reset::R

  # function that computes theta
  gettheta::T

  # function that computes the Bayes Factor surprise
  getsbf::S

  # function that returns the posterior
  getposterior::P

  # function that updates the current parameters
  update::U  

  # flag that indicates whether to update all parameters
  # or just the current column
  updateallcols::Bool

  # name of the update rule (useful for graphs)
  str
end



