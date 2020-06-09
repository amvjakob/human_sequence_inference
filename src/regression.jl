using GLM, Printf, Dates, JuliennedArrays

include("decode.jl")
include("utils.jl")
include("UpdateRule.jl")

# getssh: get shanon surprise
function getssh(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Float64
  theta = rule.gettheta(x, cols)
  return -log(theta) / log(2.0)
end

function decodessh(seq::Array{<:Integer,1},
                   rule::UpdateRule)
  # define callback to compute Shannon surprises
  callback = Callback(getssh, Float64)

  # decode sequence
  return decode(seq, rule, callback)
end



function getsbftheta(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Tuple{Float64,Float64}
  sbf   = rule.getsbf(x, cols)
  theta = rule.gettheta(x, cols)
  return (sbf, theta)
end

function decodesbftheta(seq::Array{<:Integer,1},
                        rule::UpdateRule)
  # define callback to compute Bayes Factor surprise and theta
  callback = Callback(getsbftheta, Tuple{Float64,Float64})

  # decode sequence
  result = decode(seq, rule, callback)
  return map(r -> r[1], result), map(r -> r[2], result)
end



function getposterior(rule::UpdateRule, x::Integer, cols::Array{<:Integer,1})::Array{Float64,1}
  return rule.getposterior() |> copy # params()[2]
end

function decodeposterior(seq::Array{<:Integer,1},
                         rule::UpdateRule)
  # define callback to compute posterior
  callback = Callback(getposterior, Array{Float64,1})

  # decode sequence
  return decode(seq, rule, callback)
end

function onehotencode(shape, index)
  onehot = zeros(shape...)
  onehot[:,index] .= 1
  return onehot
end

function onehotencode_ssh(s::Tuple{Int,Array{Float64,1}})
  block, ssh = s

  onehot_shape = (length(ssh), 4)
  onehot = onehotencode(onehot_shape, block)

  return hcat(onehot, ssh)
end

function regression(subject, sensor, time, rule::UpdateRule)
  # get suprises per block
  # shape is [n_block] -> [n_elements_in_block]
  ssh = decodessh.(subject.seq, rule)
  ssh = map((s,i) -> s[i], ssh, subject.seqIdx)

  # one-hot encode surprises
  ssh_onehot = map(onehotencode_ssh, enumerate(ssh))
  #ssh_onehot = map(v -> hcat(ones(size(v)...), v), ssh)

  # get megs per block
  megs = map(m -> m[:,sensor,time], subject.meg)

  # calc regression
  X = reduce(vcat, ssh_onehot)
  y = reduce(vcat, megs)

  ols = fit(LinearModel, X, y);

  return ssh, megs, ols, y
end


function mapssh(subject, models, onehotencode_blocks = false, verbose = 0)
  # init rule decode results
  ssh = Array{Array{Array{Float64,2},1},1}(undef, length(models))

  # decode sequence for every rule
  verbose > 1 && lg("mapsubject: decoding sequence for all models")

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
;
