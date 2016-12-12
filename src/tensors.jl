# handy operator ⊙ (\odot)
⊙ = contract

# abstract type for multi-dispatching
abstract AbstractTensor{T,N} <: AbstractArray{T,N}

"""
Blocked Sparse Symmetric pure n-th Order Tensor
"""
immutable BSSTensor{T<:Real,N,Order} <: AbstractTensor{T,N}
    block::Array{T,N}
    index::Vector{NTuple{N,Int}}
    dims::NTuple{Order,Int}
end

Base.nnz(A::BSSTensor) = length(A.index)
Base.size(A::BSSTensor) = A.dims
Base.size(A::BSSTensor, i::Integer) = A.dims[i]
Base.length(A::BSSTensor) = prod(A.dims)
Base.getindex{T<:Real}(A::BSSTensor{T,2,4}, i::Int, a::Int, j::Int, b::Int) = A.block[a,b]
Base.getindex{T<:Real}(A::BSSTensor{T,3,6}, i::Int, a::Int, j::Int, b::Int, k::Int, c::Int) = A.block[a,b,c]

function contract{T<:Real}(𝐇::BSSTensor{T,2,4}, 𝐱::Matrix{T})
    pixelNum, labelNum = size(𝐇,1), size(𝐇,2)
    𝐌 = zeros(T, pixelNum, labelNum)
    @inbounds for n in 1:nnz(𝐇)
        i, j = 𝐇.index[n]
        for ll in CartesianRange(size(𝐇.block))
            a, b = ll.I
            𝐌[i,a] += 𝐇[i,a,j,b] * 𝐱[j,b]
            𝐌[j,b] += 𝐇[i,a,j,b] * 𝐱[i,a]
        end
    end
    return reshape(𝐌, pixelNum*labelNum)
end

function contract{T<:Real}(𝐇::BSSTensor{T,3,6}, 𝐱::Matrix{T})
    pixelNum, labelNum = size(𝐇,1), size(𝐇,2)
    𝐌 = zeros(T, pixelNum, labelNum)
    @inbounds for n in 1:nnz(𝐇)
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
Sparse Symmetric pure n-th Order Tensor
"""
immutable SSTensor{T<:Real,Order} <: AbstractTensor{T,Order}
    data::Vector{T}
    index::Vector{NTuple{Order,Int}}
    dims::NTuple{Order,Int}
end

Base.nnz(A::SSTensor) = length(A.data)
Base.size(A::SSTensor) = A.dims
Base.size(A::SSTensor, i::Integer) = A.dims[i]
Base.length(A::SSTensor) = prod(A.dims)

function contract{T<:Real}(𝐇::SSTensor{T,2}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝐇,1))
    @inbounds for i in 1:nnz(𝐇)
        x, y = 𝐇.index[i]
        𝐯[x] += 𝐇.data[i] * 𝐱[y]
        𝐯[y] += 𝐇.data[i] * 𝐱[x]
    end
    return 𝐯
end

function contract{T<:Real}(𝐇::SSTensor{T,3}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝐇,1))
    @inbounds for i in 1:nnz(𝐇)
        x, y, z = 𝐇.index[i]
        𝐯[x] += 2.0 * 𝐇.data[i] * 𝐱[y] * 𝐱[z]
        𝐯[y] += 2.0 * 𝐇.data[i] * 𝐱[x] * 𝐱[z]
        𝐯[z] += 2.0 * 𝐇.data[i] * 𝐱[x] * 𝐱[y]
    end
    return 𝐯
end
