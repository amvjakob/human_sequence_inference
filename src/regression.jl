using GLM, Printf, Dates, JuliennedArrays

include("decode.jl")
include("utils.jl")
include("UpdateRule.jl")

function compute_shannon_surprise(rule, x, col)
  theta = rule.gettheta(col, x)
  return -log(theta) / log(2.0)
end

function decode_shannon_surprise(seq::Array{Integer,1},
                                 m::Integer,
                                 alpha_0::Array{Float64,2}, 
                                 rule::UpdateRule,
                                 ignore_partial_window = true,
                                 N = 2)

  # define callback to compute Shannon surprises
  callback = Callback(compute_shannon_surprise, Float64)

  # decode sequence
  surprises = decode(seq, m, alpha_0, rule,
                     callback = callback,
                     ignore_partial_window = ignore_partial_window,
                     N = N)

  if ignore_partial_window && m > 0
    surprises[1:m] .= -log(1.0 / N) / log(2.0)
  end

  return surprises # [2:end]
end


function onehotencode(shape, index)
  onehot = zeros(shape...)
  onehot[:,index] .= 1
  return onehot
end

function onehotencode_surprise(s::Tuple{Int,Array{Float64,1}})
  block, surprise = s

  onehot_shape = (length(surprise), 4)
  onehot = onehotencode(onehot_shape, block)

  return hcat(onehot, surprise)
end

function regression(subject, sensor, time, rule, m, alpha_0)
  # get suprises per block
  surprises = decode_shannon_surprise.(subject.seq, m, Ref(alpha_0), Ref(rule))
  surprises = map((s,i) -> s[i], surprises, subject.seqIdx)

  # one-hot encode surprises
  surprises_onehot = map(onehotencode_surprise, enumerate(surprises))
  #surprises_onehot = map(v -> hcat(ones(size(v)...), v), surprises)

  # get megs per block
  megs = map(m -> m[:,sensor,time], subject.meg)

  # calc regression
  X = reduce(vcat, surprises_onehot)
  y = reduce(vcat, megs)

  # normalize y
  #my = mean(y)
  #sy = std(y)
  #y = (y .- my) ./ sy

  ols = fit(LinearModel, X, y);

  #megs = map(m -> (m .- my) ./ sy, megs)
  return surprises, megs, ols, y
end

function mapSubject(subject, models,
                    onehotencode_blocks = false, verbose = 2)
  # init rule decode results
  surprises = Array{Array{Array{Float64,2},1},1}(undef, length(models))

  # decode sequence for every rule
  verbose > 1 && lg("mapSubject: decoding sequence for all rules")

  for (m, model) in enumerate(models)
    model_surprises = Array{Array{Float64,2},1}(undef, length(model))
    
    for (r, rule) in enumerate(model)
      # decode sequence            
      values = decode_shannon_surprise.(subject.seq, rule["m"], Ref(rule["alpha_0"]), Ref(rule["rule"]))
      # map decoded values to overlapping index
      values = map((s, idx) -> s[idx], values, subject.seqIdx)

      if onehotencode_blocks
        values = map(onehotencode_surprise, enumerate(values))
      else
        # hcat to make 2-dimensional
        values = map(v -> hcat(ones(size(v)...), v), values)
      end

      # remove block-level array structure
      @inbounds model_surprises[r] = reduce(vcat, values)
    end
    
    surprises[m] = model_surprises
  end

  # get MEG values
  verbose > 1 && lg("mapSubject: decoding MEG")
  megs = reduce(vcat, subject.meg)

  return surprises, megs
end
;
