# decode.jl

using Random

include("UpdateRule.jl")
include("inference.jl")

### generate a sequence of length len containing elements between 0 and N-1
function generateseq(len; N = 2, seed = 1234)
  Random.seed!(seed)
  return rand(collect(0:N-1), len)
end


"""
    decode(seq, rule, callback) -> AbstractArray

Decode the sequence `seq` using the update rule `rule` of type
[`UpdateRule`](@ref). Optionally, a function to run on the update 
rule between each sequence elements can be passed via `callback`
(see [`Callback`](@ref)). This `callback` is called with the 
arguments `rule`, `x_t` and `cols`.

The result of applying the callback is returned. If `seq`
contains L elements, the returned array will contain L+1 elements,
as the callback is executed before, in-between and after each 
update to `rule`.
"""
function decode(seq::Array{<:Integer,1},
                rule::UpdateRule,
                callback::Callback{F,R} = Callback((_...) -> nothing, Nothing)) where {F,R}

  # signal type, N = 2 is binary
  N = length(unique(seq))
  len = length(seq)

  # reset update rule
  rule.reset()

  # window lengths
  ms = collect(0:length(rule.getposterior())-1)

  # init callback result container
  callback_result = Array{R,1}(undef, len+1)

  # decode sequence
  for t in 1:len
    x_t  = seq[t]
    # map previous sequence to column index (or 0 if previous 
    # observations are insufficiently long)
    cols = map(m -> t > m ? seq_to_col(seq[t-m:t-1], N) : 0, ms)

    # run callback and update learning rule
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

"""
    seq_to_col(seq, N) -> Integer

Transform a sequence of elements `seq` to the corresponding
number in base `N`. This number is the column index used to
retrieve the past observations in the learning rule.

For instance, the sequence `101` will be transformed to 5 
if `N` is 2.
"""
function seq_to_col(seq, N)
  # concat sequence to string
  str = join(seq)
  # we add 1 because arrays are 1-indexed
  return str == "" ? 1 : (parse(Int64, str, base = N) + 1)
end

"""
    col_to_seq(col, N) -> String

Transform the number `col` to its representation in base `N`. 
This representation corresponds to the sequence of past
observations.

For instance, the number 5 will be transformed to `"101"`
if `N` is 2.
"""
function col_to_seq(col, N)
  # col - 1 because arrays are 1-indexed
  return string(col - 1, base = N)
end
