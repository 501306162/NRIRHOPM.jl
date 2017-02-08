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
Base.getindex{T<:NTuple}(A::IndexBlock{T}, I::Vararg{Int,N}) = A.idxs[I...]
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
    # assume (i,a,j,b,k,c) indexing
    oddIdxs = I[1:2:end]
    evenIdxs = I[2:2:end]
    for i = 1:length(A.idxBlocks)
        if oddIdxs in A.idxBlocks[i]
            out = getindex(A.valBlocks[i], evenIdxs...)
        end
    end
    out
end
Base.:(==)(A::BlockedTensor, B::BlockedTensor) = A.valBlocks == B.valBlocks && A.idxBlocks == B.idxBlocks && A.dims == B.dims


# sparse tensor contraction
contract{T<:Real}(𝑻::BlockedTensor{T}, vec::Vector{T}) = reshape(𝑻 ⊙ reshape(vec,size(𝑻,1,2))), length(vec))
function contract{T<:Real}(𝑻::BlockedTensor{T}, mat::Matrix{T})
    s = zeros(mat)
    for (vals,idxs) in zip(𝑻.valBlocks, 𝑻.idxBlocks)
        s += _contract(vals, idxs, mat)
    end
    return s
end
# handy operator ⊙ (\odot)
⊙ = contract

function _contract{T<:Real,2}(vals::ValueBlock{T,2}, idxs::IndexBlock{NTuple{2,Int}}, mat::Matrix{T})
    s = zeros(mat)
    for (i,j) in idxs
        for 𝒊 in CartesianRange(size(vals))
            a, b = 𝒊.I
            s[i,a] += vals[a,b] * mat[j,b]
            s[j,b] += vals[a,b] * mat[i,a]
        end
    end
    return s
end

function _contract{T<:Real,3}(vals::ValueBlock{T,3}, idxs::IndexBlock{NTuple{3,Int}}, mat::Matrix{T})
    s = zeros(mat)
    for (i,j,k) in idxs
        for 𝒊 in CartesianRange(size(vals))
            a, b, c = 𝒊.I
            s[i,a] += 2.0 * vals[a,b,c] * mat[j,b] * mat[k,c]
            s[j,b] += 2.0 * vals[a,b,c] * mat[i,a] * mat[k,c]
            s[k,c] += 2.0 * vals[a,b,c] * mat[i,a] * mat[j,b]
        end
    end
    return s
end

function _contract{T<:Real,4}(vals::ValueBlock{T,4}, idxs::IndexBlock{NTuple{4,Int}}, mat::Matrix{T})
    s = zeros(mat)
    for (i, j, k, m) in idxs, 𝒊 in CartesianRange(size(vals))
            a, b, c, d = 𝒊.I
            s[i,a] += 6.0 * vals[a,b,c,d] * mat[j,b] * mat[k,c] * mat[m,d]
            s[j,b] += 6.0 * vals[a,b,c,d] * mat[i,a] * mat[k,c] * mat[m,d]
            s[k,c] += 6.0 * vals[a,b,c,d] * mat[i,a] * mat[j,b] * mat[m,d]
            s[m,d] += 6.0 * vals[a,b,c,d] * mat[i,a] * mat[j,b] * mat[k,c]
        end
    end
    return s
end
