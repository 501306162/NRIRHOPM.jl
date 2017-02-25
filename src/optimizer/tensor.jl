abstract AbstractSymmetricSparseTensor{T,N} <: AbstractArray{T,N}
abstract AbstractTensorBlock{T,N} <: AbstractArray{T,N}


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


# todo: BlockedTensor{Tv<:Real,N,Ti<:NTuple{N,Int},Order}
immutable BlockedTensor{Tv<:Real,N,Ti<:NTuple,Order} <: AbstractSymmetricSparseTensor{Tv,Order}
    valBlocks::Vector{ValueBlock{Tv,N}}
    idxBlocks::Vector{IndexBlock{Ti}}
    dims::NTuple{Order,Int}
end
Base.size(A::BlockedTensor) = A.dims
function Base.getindex{Tv<:Real,N,Ti<:NTuple,Order}(A::BlockedTensor{Tv,N,Ti,Order}, I::Vararg{Int,Order})
    # this method iterates over all idxBlocks to query the result,
    # which is very slow and should not be used in performance-sensitive code.
    # the use case is playing or testing small BlockedTensors in REPL, where Base.show()
    # will automatically print the full tensor. however, things turn out so evil
    # when dealing large BlockedTensors, one should add `;` in the end of the expression
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
function Base.full{Tv<:Real,N,Ti<:NTuple,Order}(A::BlockedTensor{Tv,N,Ti,Order})
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
Base.:(==)(A::BlockedTensor, B::BlockedTensor) = A.valBlocks == B.valBlocks && A.idxBlocks == B.idxBlocks && A.dims == B.dims


# sparse tensor contraction
contract(𝑻::BlockedTensor, 𝐯::Vector) = reshape(𝑻 ⊙ reshape(𝐯,size(𝑻,1,2)), length(𝐯))
function contract(𝑻::BlockedTensor, 𝐕::Matrix)
    𝐌 = zeros(𝐕)
    for (vals,idxs) in zip(𝑻.valBlocks, 𝑻.idxBlocks)
        𝐌 += _contract(vals, idxs, 𝐕)
    end
    return 𝐌
end
# handy operator ⊙ (\odot)
⊙ = contract

function _contract{T<:Real}(vals::ValueBlock{T,2}, idxs::IndexBlock{NTuple{2,Int}}, mat::Matrix{T})
    s = zeros(mat)
    @inbounds for (i,j) in idxs, 𝒊 in CartesianRange(size(vals))
        a, b = 𝒊.I
        s[a,i] += vals[a,b] * mat[b,j]
        s[b,j] += vals[a,b] * mat[a,i]
    end
    return s
end

function _contract{T<:Real}(vals::ValueBlock{T,3}, idxs::IndexBlock{NTuple{3,Int}}, mat::Matrix{T})
    s = zeros(mat)
    @inbounds for (i,j,k) in idxs, 𝒊 in CartesianRange(size(vals))
        a, b, c = 𝒊.I
        s[a,i] += 2.0 * vals[a,b,c] * mat[b,j] * mat[c,k]
        s[b,j] += 2.0 * vals[a,b,c] * mat[a,i] * mat[c,k]
        s[c,k] += 2.0 * vals[a,b,c] * mat[a,i] * mat[b,j]
    end
    return s
end

function _contract{T<:Real}(vals::ValueBlock{T,4}, idxs::IndexBlock{NTuple{4,Int}}, mat::Matrix{T})
    s = zeros(mat)
    @inbounds for (i, j, k, m) in idxs, 𝒊 in CartesianRange(size(vals))
        a, b, c, d = 𝒊.I
        s[a,i] += 6.0 * vals[a,b,c,d] * mat[b,j] * mat[c,k] * mat[d,m]
        s[b,j] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[c,k] * mat[d,m]
        s[c,k] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[b,j] * mat[d,m]
        s[d,m] += 6.0 * vals[a,b,c,d] * mat[a,i] * mat[b,j] * mat[c,k]
    end
    return s
end
