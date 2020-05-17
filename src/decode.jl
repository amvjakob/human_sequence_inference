# decode.jl

using Random

include("UpdateRule.jl")
include("inference.jl")

### Generate a sequence of length len containing elements between 0 and N-1
function generateSeq(len, N = 2, seed = 1234)
  Random.seed!(seed)
  return rand(collect(0:N-1), len)
end


### Decode a sequence

# ignoreFirstM: flag that indicates the action to take when the previous
#   sequence is not yet m elements long
#   ignore or average over possible previous sequences
function decode(seq::Array{<:Integer,1},
                m::Integer,
                alpha_0::Array{Float64,2}, 
                rule::UpdateRule,
                callback::Callback{F,R} = Callback((_...) -> nothing, Nothing),
                ignore_partial_window = true,
                N = 2) where {F,R}

  len = length(seq)
  d = 2^m

  # check for correct dimensions
  @assert(size(alpha_0) == (2, d))

  # init update rule
  rule.init(alpha_0)

  # callback result
  callback_result = Array{R,1}(undef, len+1)

  # partial window
  if !ignore_partial_window
    for t in 1:m
      x_t = seq[t]

      # we have some missing elements at the beginning of the window
      # we solve this by averaging over possible value
      mincol = t > 1 ? seq_to_col(seq[1:t-1], N) : 1
      step = 2^(t-1)
      maxcol = d

      # perform callback
      callback_result[t] = callback.run(rule, x_t, mincol)

      # add pseudocount of 1 to every possible value
      for col in mincol:step:maxcol
        rule.update(x_t, col)
      end
    end
  end

  # full window
  for t in m+1:len
    x_t = seq[t]
    col = seq_to_col(seq[t-m:t-1], N)

    callback_result[t] = callback.run(rule, x_t, col)
    rule.update(x_t, col)
  end

  # add final callback result
  x_t = seq[end]
  col = seq_to_col(seq[end-m:end-1], N)
  callback_result[end] = callback.run(rule, x_t, col)

  return callback_result
end

### Decode a sequence for a model that performs inference over m
function decode_with_inference(seq::Array{<:Integer,1},
                               rule::UpdateRule,
                               prior::Array{Float64,1},
                               callback::Callback{F,R} = Callback((_...) -> nothing, Nothing),
                               N = 2) where {F,R}
  len = length(seq)
  ms  = collect(0:length(prior)-1)

  # init update rule
  rule.init(prior, N)

  # callback result
  callback_result = Array{R,1}(undef, len+1)

  # decode sequence
  for t in 1:len
    x_t  = seq[t]
    cols = map(m -> t > m ? seq_to_col(seq[t-m:t-1], N) : 0, ms)

    callback_result[t] = callback.run(rule, x_t, cols)
    rule.update(x_t, cols)
  end

  # add final callback result
  x_t = seq[len]
  cols = map(m -> len > m ? seq_to_col(seq[len-m:len-1], N) : 0, ms)
  callback_result[end] = callback.run(rule, x_t, cols)

  return callback_result
end



function seq_to_col(seq, N)
  # concat sequence to string
  str = join(seq)
  return str == "" ? 1 : (parse(Int64, str, base = N) + 1)
end

function col_to_seq(col, N)
  # returns a string
  return string(col - 1, base = N)
end
