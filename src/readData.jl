using MAT, Printf

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


mutable struct SubjectData{S,SeqIdx,M,MegIdx,T,MegData}
    # data
    seq::S
    seqIdx::SeqIdx
    meg::M
    megIdx::MegIdx
    time::T
    megData::MegData
end


function getNValidTrials(s::SubjectData)
    return reduce(+, map(m -> size(m, 1), s.meg))
end

function getTimestamps(s::SubjectData)
    return s.time
end

function getSeqForBlock(s::SubjectData, block)
    return s.seq[block]
end

function getMEGForBlock(s::SubjectData, block)
    return s.meg[block]
end

getFilename(subjectNumber) = @sprintf(
    "../../../../../../Documents/human_sequence_inference_data/subject%s.mat",
    lpad(subjectNumber, 2, "0")
)

function loadSubjectData(filename)
    file = matopen(filename)

    # read file
    seq = read(file, "seq")
    seqIdx = read(file, "seqidx")
    megData = read(file, "meg")

    close(file)

    # transform sequence
    seqRaw = dropdims(map(s -> s[1,:], seq), dims=1);
    seqCleanedGlobalIdx = findnonnan.(seqRaw);
    seqCleaned = map((s,i) -> Int.(s[i] .- 1), seqRaw, seqCleanedGlobalIdx);

    # transform meg data
    megTrialInfo = Int.(megData["trialinfo"])
    megCleaned = Array{Array{Float64,3},1}(undef, length(seqRaw))
    @inbounds for i in eachindex(seqRaw)
        megCleaned[i] = megData["trial"][megTrialInfo[:,3] .=== i,:,:]
    end
    megGlobalIdx = map(s -> Int.(s[1,:]), seqIdx)

    # compute overlapping indices
    seqCleanedIdx = map((seqIdx, megIdx) -> findall(in(megIdx), seqIdx), seqCleanedGlobalIdx, megGlobalIdx)
    megCleanedIdx = map((seqIdx, megIdx) -> findall(in(seqIdx), megIdx), seqCleanedGlobalIdx, megGlobalIdx)

    # apply overlap index to MEG and make units pT
    megCleaned = map((m,i) -> 1e12 * m[i,:,:], megCleaned, megCleanedIdx)

    return SubjectData(
        seqCleaned,
        seqCleanedIdx,
        megCleaned,
        megCleanedIdx,
        megData["time"][1,:],
        megData
    )
end
