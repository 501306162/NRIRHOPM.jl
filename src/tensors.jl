# abstract type for multi-dispatching
abstract AbstractTensor{T,N} <: AbstractArray{T,N}

"""
Tensor Block
"""
immutable TensorBlock{T<:Real,N,Order} <: AbstractArray{T,N}
    block::Array{T,N}
    index::Vector{NTuple{N,Int}}
    dims::NTuple{Order,Int}
end

Base.nnz(𝐇::TensorBlock) = length(𝐇.index)
Base.size(𝐇::TensorBlock) = 𝐇.dims
Base.size(𝐇::TensorBlock, i::Integer) = 𝐇.dims[i]
Base.length(𝐇::TensorBlock) = prod(𝐇.dims)
Base.getindex{T<:Real}(𝐇::TensorBlock{T,2,4}, i::Integer, a::Integer, j::Integer, b::Integer) = 𝐇.block[a,b]
Base.getindex{T<:Real}(𝐇::TensorBlock{T,3,6}, i::Integer, a::Integer, j::Integer, b::Integer, k::Integer, c::Integer) = 𝐇.block[a,b,c]
==(x::TensorBlock, y::TensorBlock) = x.block == y.block && x.index == y.index && x.dims == y.dims

function contract{T<:Real}(𝐇::TensorBlock{T,2,4}, 𝐱::Matrix{T})
    pixelNum, labelNum = size(𝐇,1), size(𝐇,2)
    𝐌 = zeros(T, pixelNum, labelNum)
    for n in 1:nnz(𝐇)
        i, j = 𝐇.index[n]
        for ll in CartesianRange(size(𝐇.block))
            a, b = ll.I
            𝐌[i,a] += 𝐇[i,a,j,b] * 𝐱[j,b]
            𝐌[j,b] += 𝐇[i,a,j,b] * 𝐱[i,a]
        end
    end
    return reshape(𝐌, pixelNum*labelNum)
end

function contract{T<:Real}(𝐇::TensorBlock{T,3,6}, 𝐱::Matrix{T})
    pixelNum, labelNum = size(𝐇,1), size(𝐇,2)
    𝐌 = zeros(T, pixelNum, labelNum)
    for n in 1:nnz(𝐇)
        i, j, k = 𝐇.index[n]
        for lll in CartesianRange(size(𝐇.block))
            a, b, c = lll.I
            𝐌[i,a] += 2.0 * 𝐇[i,a,j,b,k,c] * 𝐱[j,b] * 𝐱[k,c]
            𝐌[j,b] += 2.0 * 𝐇[i,a,j,b,k,c] * 𝐱[i,a] * 𝐱[k,c]
            𝐌[k,c] += 2.0 * 𝐇[i,a,j,b,k,c] * 𝐱[i,a] * 𝐱[j,b]
        end
    end
    return reshape(𝐌, pixelNum*labelNum)
end

"""
Blocked Sparse Symmetric pure n-th Order Tensor
"""
immutable BSSTensor{T<:Real,N,Order} <: AbstractTensor{T,N}
    blocks::Vector{TensorBlock{T,N,Order}}
    dims::NTuple{Order,Int}
end

Base.nnz(𝐇::BSSTensor) = mapreduce(nnz, +, 𝐇.blocks)
Base.size(𝐇::BSSTensor) = 𝐇.dims
Base.size(𝐇::BSSTensor, i::Integer) = 𝐇.dims[i]
Base.length(𝐇::BSSTensor) = prod(𝐇.dims)
==(x::BSSTensor, y::BSSTensor) = x.blocks == y.blocks && x.dims == y.dims

function contract{T<:Real}(𝐇::BSSTensor{T}, 𝐱::Vector{T})
    pixelNum, labelNum = size(𝐇,1), size(𝐇,2)
    𝐯 = zeros(T, pixelNum*labelNum)
    for 𝐛 in 𝐇.blocks
        𝐯 += contract(𝐛, reshape(𝐱, pixelNum, labelNum))
    end
    return 𝐯
end

"""
Sparse Symmetric pure n-th Order Tensor
"""
immutable SSTensor{T<:Real,Order} <: AbstractTensor{T,Order}
    data::Vector{T}
    index::Vector{NTuple{Order,Int}}
    dims::NTuple{Order,Int}
end

Base.nnz(𝐇::SSTensor) = length(𝐇.data)
Base.size(𝐇::SSTensor) = 𝐇.dims
Base.size(𝐇::SSTensor, i::Integer) = 𝐇.dims[i]
Base.length(𝐇::SSTensor) = prod(𝐇.dims)

function contract{T<:Real}(𝐇::SSTensor{T,2}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝐇,1))
    for i in 1:nnz(𝐇)
        x, y = 𝐇.index[i]
        𝐯[x] += 𝐇.data[i] * 𝐱[y]
        𝐯[y] += 𝐇.data[i] * 𝐱[x]
    end
    return 𝐯
end

function contract{T<:Real}(𝐇::SSTensor{T,3}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝐇,1))
    for i in 1:nnz(𝐇)
        x, y, z = 𝐇.index[i]
        𝐯[x] += 2.0 * 𝐇.data[i] * 𝐱[y] * 𝐱[z]
        𝐯[y] += 2.0 * 𝐇.data[i] * 𝐱[x] * 𝐱[z]
        𝐯[z] += 2.0 * 𝐇.data[i] * 𝐱[x] * 𝐱[y]
    end
    return 𝐯
end

# handy operator ⊙ (\odot)
⊙ = contract
