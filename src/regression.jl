# regression.jl

using GLM

include("decode.jl")
include("utils.jl")
include("UpdateRule.jl")

"""
    getssh(rule, x, cols) -> Float64

Compute the Shannon surprise of the learning rule `rule` if `x`
were observed, given the previous observations encoded in `cols`.
"""
function getssh(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Float64
  theta = rule.gettheta(x, cols)
  return -log(theta) / log(2.0)
end

"""
    decodessh(seq, rule)

Decode the sequence `seq` using the learning rule `rule`, and 
compute the Shannon surprise for each observation.
"""
function decodessh(seq::Array{<:Integer,1},
                   rule::UpdateRule)
  # define callback to compute Shannon surprises
  callback = Callback(getssh, Float64)

  # decode sequence
  return decode(seq, rule, callback)
end

"""
    getsbftheta(rule, x, cols) -> Tuple{Float64,Float64}

Compute the surprise Bayes Factor and the expected value of theta
of the learning rule `rule` if `x` were observed, given the previous
observations encoded in `cols`.
"""
function getsbftheta(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Tuple{Float64,Float64}
  sbf   = rule.getsbf(x, cols)
  theta = rule.gettheta(x, cols)
  return (sbf, theta)
end

"""
    decodesbftheta(seq, rule)

Decode the sequence `seq` using the learning rule `rule`, and 
compute the Bayes Factor surprise and the expected value of theta
for each observation.
"""
function decodesbftheta(seq::Array{<:Integer,1},
                        rule::UpdateRule)
  # define callback to compute Bayes Factor surprise and theta
  callback = Callback(getsbftheta, Tuple{Float64,Float64})

  # decode sequence
  result = decode(seq, rule, callback)
  return map(r -> r[1], result), map(r -> r[2], result)
end


"""
    getposterior(rule, x, cols) -> Array{Float64,1}

Compute the posterior probability of different window lengths of
the learning rule `rule` if `x` were observed, given the previous
observations encoded in `cols`.
"""
function getposterior(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Array{Float64,1}
  return rule.getposterior() |> copy
end

"""
    decodeposterior(seq, rule)

Decode the sequence `seq` using the learning rule `rule`, and 
compute the posterior at each observation.
"""
function decodeposterior(seq::Array{<:Integer,1},
                         rule::UpdateRule)
  # define callback to compute posterior
  callback = Callback(getposterior, Array{Float64,1})

  # decode sequence
  return decode(seq, rule, callback)
end


"""
    onehotencode(shape, index)

Create a 2-D Matrix of zeros with the given `shape`, and set 1s
for the whole column given by `index`.
"""
function onehotencode(shape, index)
  onehot = zeros(shape...)
  onehot[:,index] .= 1
  return onehot
end

"""
    onehotencode_ssh(s::Tuple{Int,Array{Float64,1}})

Transforms a tuple of a block index and an array of surprise values
to a 2-D Matrix of shape `(length(Array), 5)`, where the 4 first 
columns contain the one-hot-encoded value of the block index, 
and the 5th column contains the surprise values.
"""
function onehotencode_ssh(s::Tuple{Int,Array{Float64,1}})
  block, ssh = s

  onehot_shape = (length(ssh), 4)
  onehot = onehotencode(onehot_shape, block)

  return hcat(onehot, ssh)
end


"""
    regression(subject, sensor, time, rule) -> ssh, megs, ols, y

Performs a linear regression on subject MEG data `subject` (see
[`SubjectData`](@ref)) at sensor `sensor` and time `time`. Uses
the learning rule `rule`.
"""
function regression(subject, sensor, time, rule::UpdateRule)
  # get suprises per block
  # shape is [n_block] -> [n_elements_in_block]
  ssh = decodessh.(subject.seq, Ref(rule))
  ssh = map((s,i) -> s[i], ssh, subject.seqIdx)

  # one-hot encode surprises
  ssh_onehot = map(onehotencode_ssh, enumerate(ssh))
  # alternatively, don't one-hot-encode by block
  #ssh_onehot = map(v -> hcat(ones(size(v)...), v), ssh)

  # get MEG data per block
  megs = map(m -> m[:,sensor,time], subject.meg)

  # perform linear regression
  X = reduce(vcat, ssh_onehot)
  y = reduce(vcat, megs)

  ols = fit(LinearModel, X, y)

  return ssh, megs, ols, y
end


"""
    mapssh(subject, models, onehotencode_blocks, verbose)

Map the subject MEG data given in `subject` (see [`SubjectData`](@ref))
to Shannon surprise for each learning rule given in `models`. Control
whether to one-hot-encode the surprise values using `onehotencode_blocks`.
"""
function mapssh(subject, models, onehotencode_blocks = false, verbose = 0)
  # init rule decode results
  ssh = Array{Array{Array{Float64,2},1},1}(undef, length(models))

  # decode sequence for every rule
  verbose > 1 && lg("mapssh: decoding sequence for all models")

  for (m, model) in enumerate(models)
    modelssh = Array{Array{Float64,2},1}(undef, length(model))
    
    for (r, rule) in enumerate(model)
      # decode sequence            
      values = decodessh.(subject.seq, Ref(rule))
      # map decoded values to overlapping index
      values = map((s, idx) -> s[idx], values, subject.seqIdx)

      if onehotencode_blocks
        values = map(onehotencode_ssh, enumerate(values))
      else
        # hcat to make 2-dimensional
        values = map(v -> hcat(ones(size(v)...), v), values)
      end

      # remove block-level array structure
      modelssh[r] = reduce(vcat, values)
    end
    
    ssh[m] = modelssh
  end

  # get MEG values (remove block-level array structure)
  megs = reduce(vcat, subject.meg)

  return ssh, megs
end


"""
    mapposterior(subject, models, verbose)

Map the subject MEG data given in `subject` (see [`SubjectData`](@ref))
to posterior probability of different window lengths for each learning
rule given in `models`.
"""
function mapposterior(subject, models, verbose = 0)
  # init rule decode results
  posterior = Array{Array{Array{Float64,2},1},1}(undef, length(models))

  # decode sequence for every rule
  verbose > 1 && lg("mapposterior: decoding sequence for all models")

  for (m, model) in enumerate(models)
    modelposterior = Array{Array{Float64,2},1}(undef, length(model))
    
    for (r, rule) in enumerate(model)
      # decode sequence            
      values = decodeposterior.(subject.seq, Ref(rule))
      # map decoded values to overlapping index
      values = map((s, idx) -> s[idx], values, subject.seqIdx)

      # remove block-level array structure
      modelposterior[r] = reduce(vcat, values) |> unwrap
    end
    
    posterior[m] = modelposterior
  end

  return posterior
end
