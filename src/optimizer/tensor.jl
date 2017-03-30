abstract AbstractSymmetricSparseTensor{T,N} <: AbstractArray{T,N}
abstract AbstractTensorBlock{T,N} <: AbstractArray{T,N}

immutable BlockedTensor{Tv<:Real,N,Ti<:NTuple,Order} <: AbstractSymmetricSparseTensor{Tv,Order}
    vals::Array{Tv,N}
    idxs::Vector{Ti}
    dims::NTuple{Order,Int}
end
Base.:(==)(A::BlockedTensor, B::BlockedTensor) = A.vals == B.vals && A.idxs == B.idxs && A.dims == B.dims
function Base.getindex{Tv<:Real,N,Ti<:NTuple,Order}(A::BlockedTensor{Tv,N,Ti,Order}, I::Vararg{Int,Order})
    # do NOT use this method in performance-sensitive code
    out = zero(Tv)
    # assume (a,i,b,j,c,k) indexing
    oddIdxs = I[1:2:end]
    evenIdxs = I[2:2:end]
    if evenIdxs in A.idxs
        out = getindex(A.vals, oddIdxs...)
    end
    out
end


immutable ValueBlock{T<:Real,N} <: AbstractTensorBlock{T,N}
    vals::Array{T,N}
end
Base.size(A::ValueBlock) = size(A.vals)
Base.getindex(A::ValueBlock, i::Integer) = A.vals[i]
Base.getindex{T<:Real,N}(A::ValueBlock{T,N}, I::Vararg{Int,N}) = A.vals[I...]
Base.:(==)(A::ValueBlock, B::ValueBlock) = A.vals == B.vals

# todo: IndexBlock{N,T<:NTuple{N,Int}} -- this is the so called triangular dispatch, which will be supported on julia-v0.6+.
immutable IndexBlock{T<:NTuple} <: AbstractTensorBlock{T,1}
    idxs::Vector{T}
end
Base.size(A::IndexBlock) = size(A.idxs)
Base.getindex(A::IndexBlock, i::Integer) = A.idxs[i]
Base.getindex(A::IndexBlock, I...) = A.idxs[I...]
Base.:(==)(A::IndexBlock, B::IndexBlock) = A.idxs == B.idxs

# todo: CompositeBlockedTensor{Tv<:Real,N,Ti<:NTuple{N,Int},Order}
immutable CompositeBlockedTensor{Tv<:Real,N,Ti<:NTuple,Order} <: AbstractSymmetricSparseTensor{Tv,Order}
    valBlocks::Vector{ValueBlock{Tv,N}}
    idxBlocks::Vector{IndexBlock{Ti}}
    dims::NTuple{Order,Int}
end
function Base.getindex{Tv<:Real,N,Ti<:NTuple,Order}(A::CompositeBlockedTensor{Tv,N,Ti,Order}, I::Vararg{Int,Order})
    # this method iterates over all idxBlocks to query the result,
    # which is very slow and should not be used in performance-sensitive code.
    # the use case is playing or testing small CompositeBlockedTensors in REPL, where Base.show()
    # will automatically print the full tensor. however, things turn out so evil
    # when dealing large CompositeBlockedTensors, one should add `;` in the end of the expression
    # to mute printing.
    out = zero(Tv)
    # assume (a,i,b,j,c,k) indexing
    oddIdxs = I[1:2:end]
    evenIdxs = I[2:2:end]
    for i = 1:length(A.idxBlocks)
        if evenIdxs in A.idxBlocks[i]
            out = getindex(A.valBlocks[i], oddIdxs...)
        end
    end
    out
end
Base.:(==)(A::CompositeBlockedTensor, B::CompositeBlockedTensor) = A.valBlocks == B.valBlocks && A.idxBlocks == B.idxBlocks && A.dims == B.dims


Base.size(A::AbstractSymmetricSparseTensor) = A.dims
function Base.full{Tv<:Real,Order}(A::AbstractSymmetricSparseTensor{Tv,Order})
    B = reshape(A, ntuple(x->prod(size(A,1,2)), Int(Order/2)))
    S = zeros(B)
    for 𝒊 in CartesianRange(size(B))
        if B[𝒊] != 0
            for idx in permutations(𝒊)
                S[idx...] = B[𝒊]
            end
        end
    end
    return reshape(S, size(A))
end


# sparse tensor contraction
function contract{Tv<:Real,N,Ti<:NTuple}(𝑻::BlockedTensor{Tv,N,Ti,4}, 𝐕::Matrix)
    𝐌 = zeros(𝐕)
    critical = Threads.Mutex()
    Threads.@threads for idx in 𝑻.idxs
        i, j = idx
        si, sj = zeros(size(𝑻.vals,1)), zeros(size(𝑻.vals,1))
         for 𝒊 in CartesianRange(size(𝑻.vals))
            a, b = 𝒊.I
            si[a] += 𝑻.vals[a,b] * 𝐕[b,j]
            sj[b] += 𝑻.vals[a,b] * 𝐕[a,i]
        end
        lock(critical)
        𝐌[:,i] .+= si
        𝐌[:,j] .+= sj
        unlock(critical)
    end
    return 𝐌
end

function contract{Tv<:Real,N,Ti<:NTuple}(𝑻::BlockedTensor{Tv,N,Ti,6}, 𝐕::Matrix)
    𝐌 = zeros(𝐕)
    critical = Threads.Mutex()
    Threads.@threads for idx in 𝑻.idxs
        i, j, k = idx
        si, sj, sk = zeros(size(𝑻.vals,1)), zeros(size(𝑻.vals,1)), zeros(size(𝑻.vals,1))
        for 𝒊 in CartesianRange(size(𝑻.vals))
            a, b, c = 𝒊.I
            si[a] += 2.0 * 𝑻.vals[a,b,c] * 𝐕[b,j] * 𝐕[c,k]
            sj[b] += 2.0 * 𝑻.vals[a,b,c] * 𝐕[a,i] * 𝐕[c,k]
            sk[c] += 2.0 * 𝑻.vals[a,b,c] * 𝐕[a,i] * 𝐕[b,j]
        end
        lock(critical)
        𝐌[:,i] .+= si
        𝐌[:,j] .+= sj
        𝐌[:,k] .+= sk
        unlock(critical)
    end
    return 𝐌
end


function contract(𝑻::CompositeBlockedTensor, 𝐕::Matrix)
    𝐌 = zeros(𝐕)
    for (vals,idxs) in zip(𝑻.valBlocks, 𝑻.idxBlocks)
        _contract!(𝐌, vals, idxs, 𝐕)
    end
    return 𝐌
end

function _contract!{T<:Real}(s::Matrix{T}, vals::ValueBlock{T,2}, idxs::IndexBlock{NTuple{2,Int}}, mat::Matrix{T})
    for idx in idxs
        i, j = idx
        @inbounds for 𝒊 in CartesianRange(size(vals))
            a, b = 𝒊.I
            s[a,i] += vals[a,b] * mat[b,j]
            s[b,j] += vals[a,b] * mat[a,i]
        end
    end
end

function _contract!{T<:Real}(s::Matrix{T}, vals::ValueBlock{T,3}, idxs::IndexBlock{NTuple{3,Int}}, mat::Matrix{T})
    for idx in idxs
        i, j, k = idx
        @inbounds for 𝒊 in CartesianRange(size(vals))
            a, b, c = 𝒊.I
            s[a,i] += 2.0 * vals[a,b,c] * mat[b,j] * mat[c,k]
            s[b,j] += 2.0 * vals[a,b,c] * mat[a,i] * mat[c,k]
            s[c,k] += 2.0 * vals[a,b,c] * mat[a,i] * mat[b,j]
        end
    end
end

function _contract!{T<:Real}(s::Matrix{T}, vals::ValueBlock{T,4}, idxs::IndexBlock{NTuple{4,Int}}, mat::Matrix{T})
    for idx in idxs
        i, j, k, m = idx
        @inbounds for 𝒊 in CartesianRange(size(vals))
            a, b, c, d = 𝒊.I
            s[a,i] += 6.0 * vals[a,b,c,d] * mat[b,j] * mat[c,k] * mat[d,m]
            s[b,j] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[c,k] * mat[d,m]
            s[c,k] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[b,j] * mat[d,m]
            s[d,m] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[b,j] * mat[c,k]
        end
    end
end

contract(𝑻::AbstractSymmetricSparseTensor, 𝐯::Vector) = reshape(𝑻 ⊙ reshape(𝐯,size(𝑻,1,2)), length(𝐯))

# handy operator ⊙ (\odot)
const ⊙ = contract
