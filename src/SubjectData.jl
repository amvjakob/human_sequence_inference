# readData.jl

using MAT, Printf, FileIO, JLD2

include("utils.jl")

# in total, 4 runs of sequences were made for each subject
# each run corresponds to a block with different statistics (e.g. more alternations)

# in each run, the sequence consists of 430 elements (A or B)
# every element is ~200ms in duration, and the space between onset of x_t and onset of x_{t+1} is 1400ms
# every 12 - 18 element, no element was shown, but the subject was asked to guess the next element

# every element has a MEG measurement attached to it
# 306 sensors around the head
# 321 time points for each element, reaching from -0.25s before stimulus onset to 1s after stimulus onset

# some elements have to be discarded following pre-processing
# this is why some sequence elements don't have any associated MEG measurements

# thus, trial is N x 306 x 321, where N would be 430 * 4 = 1720 if no elements had been discarded

# trialinfo contains info about every sequence element
# -> which element index overall
# -> which element index in the block sequence
# -> which block
# -> which element

# having to remove certain trials is not a problem:
# use sequence as-is, we just don't have the MEG data of the brain response

"""
    SubjectData(seq, seqidx, meg, megidx, time, megdata)

Constructs a new structure holding subject data.
"""
struct SubjectData{S,SeqIdx,M,MegIdx,T,MegData}
  "The raw sequence."
  seq::S

  "The indices of the sequence for which there are MEG data."
  seqIdx::SeqIdx

  "The raw MEG data."
  meg::M

  "The indices of the MEG data for which there are sequence elements."
  megIdx::MegIdx
  
  "Time vector."
  time::T
  
  "Raw file data."
  megData::MegData
end


function get_n_valid_trials(s::SubjectData)
  return reduce(+, map(m -> size(m, 1), s.meg))
end

function get_timestamps(s::SubjectData)
  return s.time
end

function get_seq_for_block(s::SubjectData, block)
  return s.seq[block]
end

function get_meg_for_block(s::SubjectData, block)
  return s.meg[block]
end

function transform_mat_to_jld2(dir = "../../../../../../../Documents/human_sequence_inference_data")
  for s in 1:18
      println("subject $s")
      subject = load_subjectdata(get_filename(s, dir))
      !isdir("$dir/jld2") && mkdir("$dir/jld2")
      @save @sprintf("%s/jld2/subject%s.jld2", dir, lpad(s, 2, "0")) subject
  end
end

get_jld2_filename(subject, dir = "../../../../../../../Documents/human_sequence_inference_data/jld2") = @sprintf(
  "%s/subject%s.jld2", dir, lpad(subject, 2, "0")
)

function load_jld2(filename)
  @load filename subject
  return subject
end

get_filename(subject, dir = "../../../../../../../Documents/human_sequence_inference_data") = @sprintf(
  "%s/subject%s.mat", dir, lpad(subject, 2, "0")
)

function load_subjectdata(filename)
  file = matopen(filename)

  # read file
  seq = read(file, "seq")
  seqidx = read(file, "seqidx")
  megdata = read(file, "meg")

  close(file)

  # transform sequence
  seqraw = dropdims(map(s -> s[1,:], seq), dims=1)
  seqcleaned_globalidx = findnonnan.(seqraw)
  seqcleaned = map((s,i) -> Int.(s[i] .- 1), seqraw, seqcleaned_globalidx)

  # transform meg data
  megtrialinfo = Int.(megdata["trialinfo"])
  megcleaned = Array{Array{Float64,3},1}(undef, length(seqraw))
  for i in eachindex(seqraw)
      megcleaned[i] = megdata["trial"][megtrialinfo[:,3] .=== i,:,:]
  end
  meg_globalidx = map(s -> Int.(s[1,:]), seqidx)

  # compute overlapping indices
  seqcleaned_idx = map((seqidx, megidx) -> findall(in(megidx), seqidx), seqcleaned_globalidx, meg_globalidx)
  megcleaned_idx = map((seqidx, megidx) -> findall(in(seqidx), megidx), seqcleaned_globalidx, meg_globalidx)

  # apply overlap index to MEG
  megcleaned = map((m,i) -> m[i,:,:], megcleaned, megcleaned_idx)

  return SubjectData(
    seqcleaned,
    seqcleaned_idx,
    megcleaned,
    megcleaned_idx,
    megdata["time"][1,:],
    megdata
  )
end
