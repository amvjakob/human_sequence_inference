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

function decode(seq::Array{<:Integer,1},
                rule::UpdateRule,
                callback::Callback{F,R} = Callback((_...) -> nothing, Nothing)) where {F,R}

  # signal type, N = 2 is binary
  N = length(unique(seq))

  # reset update rule
  rule.reset()

  len = length(seq)
  ms  = collect(0:length(rule.getposterior())-1)

  # init callback result container
  callback_result = Array{R,1}(undef, len+1)

  # partial window
  """
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
  """;

  # decode sequence
  for t in 1:len
    x_t  = seq[t]
    cols = map(m -> t > m ? seq_to_col(seq[t-m:t-1], N) : 0, ms)

    callback_result[t] = callback.run(rule, x_t, cols)
    rule.update(x_t, cols)
  end

  # add final callback result
  x_t  = seq[len]
  cols = map(m -> len > m ? seq_to_col(seq[len-m:len-1], N) : 0, ms)
  callback_result[len+1] = callback.run(rule, x_t, cols)

  return callback_result
end

### helper functions

# transform a sequence to a column index in the base N
function seq_to_col(seq, N)
  # concat sequence to string
  str = join(seq)
  return str == "" ? 1 : (parse(Int64, str, base = N) + 1)
end

# transform a column index in the base N to the corresponding sequence
function col_to_seq(col, N)
  # returns a string
  return string(col - 1, base = N)
end
