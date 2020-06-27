# UpdateRule.jl


"""
    Callback(fn::F, returntype::R)

Creates a new callback based on the function `fn` expected to
return elements of type `returntype`.
"""
struct Callback{F,R}
  """
    run(rule, x, cols) -> R

  Run the callback function on the update rule `rule`, based on
  the observation `x` and the past observations encoded in `cols`.
  """
  run::F

  function Callback(fun::F, R) where F
    return new{F,R}(fun)
  end
end


"""
    UpdateRule(reset, gettheta, getsbf, getposterior, update, updateallcols, str)

Creates a new learning rule with the given parameters.

# Arguments

- `reset()`

Reset the internal state of the learning rule to its inital value.


- `gettheta(x, cols)`

Get the expected value of the probability of observing `x` given the 
past observations contained in `cols`. `cols` is an array of column 
indices corresponding to the past observations, where each array
element corresponds to a certain window length.


- `getsbf(x, cols)`

Get the Bayes Factor surprise of observing `x` given the past
observations contained in `cols`. `cols` is an array of column 
indices corresponding to the past observations, where each array
element corresponds to a certain window length.


- `getposterior()`

Get the posterior probability over different window lengths.


- `update(x, cols)`

Update the state of the learning rule by observing `x` given the 
past observations contained in `cols`. `cols` is an array of column 
indices corresponding to the past observations, where each array
element corresponds to a certain window length.


- `updateallcols::Bool`

A flag that indicates whether to update the whole internal state or
just the state corresponding to the past observations.


- `str`

Name of the update rule.
"""
struct UpdateRule{R,T,S,P,U}
  reset::R

  gettheta::T
  getsbf::S
  getposterior::P

  update::U  

  updateallcols::Bool
  str
end
