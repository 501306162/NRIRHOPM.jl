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


immutable BlockedTensor{Tv<:Real,N,Ti<:NTuple,Order} <: AbstractSymmetricSparseTensor{Tv,Order}
    valBlocks::Vector{ValueBlock{Tv,N}}
    idxBlocks::Vector{IndexBlock{Ti}}
    dims::NTuple{Order,Int}
end
Base.size(A::BlockedTensor) = A.dims
function Base.getindex{Tv<:Real,N,Ti<:NTuple,Order}(A::BlockedTensor{Tv,N,Ti,Order}, I::Vararg{Int,Order})
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


function contract{T<:Real}(𝑻::TensorBlock{T,2,4}, 𝐗::Matrix{T})
    𝐌 = zeros(𝐗)
    for (i,j) in 𝑻.index
        for ll in CartesianRange(size(𝑻.block))
            a, b = ll.I
            𝐌[i,a] += 𝑻[a,b] * 𝐗[j,b]
            𝐌[j,b] += 𝑻[a,b] * 𝐗[i,a]
        end
    end
    return 𝐌
end

function contract{T<:Real}(𝑻::TensorBlock{T,3,6}, 𝐗::Matrix{T})
    𝐌 = zeros(𝐗)
    for (i,j,k) in 𝑻.index
        for lll in CartesianRange(size(𝑻.block))
            a, b, c = lll.I
            𝐌[i,a] += 2.0 * 𝑻[a,b,c] * 𝐗[j,b] * 𝐗[k,c]
            𝐌[j,b] += 2.0 * 𝑻[a,b,c] * 𝐗[i,a] * 𝐗[k,c]
            𝐌[k,c] += 2.0 * 𝑻[a,b,c] * 𝐗[i,a] * 𝐗[j,b]
        end
    end
    return 𝐌
end

function contract{T<:Real}(𝑻::TensorBlock{T,4,8}, 𝐗::Matrix{T})
    𝐌 = zeros(𝐗)
    for (i, j, k, m) in 𝑻.index
        for llll in CartesianRange(size(𝑻.block))
            a, b, c, d = llll.I
            𝐌[i,a] += 6.0 * 𝑻[a,b,c,d] * 𝐗[j,b] * 𝐗[k,c] * 𝐗[m,d]
            𝐌[j,b] += 6.0 * 𝑻[a,b,c,d] * 𝐗[i,a] * 𝐗[k,c] * 𝐗[m,d]
            𝐌[k,c] += 6.0 * 𝑻[a,b,c,d] * 𝐗[i,a] * 𝐗[j,b] * 𝐗[m,d]
            𝐌[m,d] += 6.0 * 𝑻[a,b,c,d] * 𝐗[i,a] * 𝐗[j,b] * 𝐗[k,c]
        end
    end
    return 𝐌
end



function contract{T<:Real}(𝑯::BSSTensor{T}, 𝐱::Vector{T})
    pixelNum, labelNum = size(𝑯,1), size(𝑯,2)
    𝐌 = zeros(T, pixelNum, labelNum)
    for 𝐛 in 𝑯.blocks
        𝐌 += contract(𝐛, reshape(𝐱, pixelNum, labelNum))
    end
    𝐯 = reshape(𝐌, pixelNum*labelNum)
end

function contract{T<:Real}(𝑯::BSSTensor{T}, 𝐗::Matrix{T})
    𝐌 = zeros(T, size(𝐗)...)
    for 𝐛 in 𝑯.blocks
        𝐌 += contract(𝐛, 𝐗)
    end
    return 𝐌
end

# handy operator ⊙ (\odot)
⊙ = contract
