abstract AbstractTensor{T,N} <: AbstractArray{T,N}

immutable TensorBlock{T<:Real,N,Order} <: AbstractTensor{T,N}
    block::Array{T,N}
    index::Vector{NTuple{N,Int}}
    dims::NTuple{Order,Int}
end

Base.size(A::TensorBlock) = size(A.block)
Base.getindex(A::TensorBlock, i::Integer) = A.block[i]
Base.getindex(A::TensorBlock, I...) = A.block[I...]
Base.:(==)(A::TensorBlock, B::TensorBlock) = A.block == B.block && A.index == B.index && A.dims == B.dims

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

"""
Blocked Sparse Symmetric pure n-th Order Tensor
"""
immutable BSSTensor{T<:Real,N,Order} <: AbstractTensor{T,N}
    blocks::Vector{TensorBlock{T,N,Order}}
    dims::NTuple{Order,Int}
end

Base.size(A::BSSTensor) = size(A.block)
Base.getindex(A::BSSTensor, i::Integer) = A.block[i]
Base.getindex(A::BSSTensor, I...) = A.block[I...]
Base.:(==)(A::BSSTensor, B::BSSTensor) = A.blocks == B.blocks && A.dims == B.dims

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

"""
Sparse Symmetric pure n-th Order Tensor
"""
immutable SSTensor{T<:Real,Order} <: AbstractTensor{T,Order}
    data::Vector{T}
    index::Vector{NTuple{Order,Int}}
    dims::NTuple{Order,Int}
end

Base.nnz(𝑯::SSTensor) = length(𝑯.data)
Base.size(𝑯::SSTensor) = 𝑯.dims
Base.size(𝑯::SSTensor, i::Integer) = 𝑯.dims[i]
Base.length(𝑯::SSTensor) = prod(𝑯.dims)

function contract{T<:Real}(𝑯::SSTensor{T,2}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝑯,1))
    for i in 1:nnz(𝑯)
        x, y = 𝑯.index[i]
        𝐯[x] += 𝑯.data[i] * 𝐱[y]
        𝐯[y] += 𝑯.data[i] * 𝐱[x]
    end
    return 𝐯
end

function contract{T<:Real}(𝑯::SSTensor{T,3}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝑯,1))
    for i in 1:nnz(𝑯)
        x, y, z = 𝑯.index[i]
        𝐯[x] += 2.0 * 𝑯.data[i] * 𝐱[y] * 𝐱[z]
        𝐯[y] += 2.0 * 𝑯.data[i] * 𝐱[x] * 𝐱[z]
        𝐯[z] += 2.0 * 𝑯.data[i] * 𝐱[x] * 𝐱[y]
    end
    return 𝐯
end

function contract{T<:Real}(𝑯::SSTensor{T,4}, 𝐱::Vector{T})
    𝐯 = zeros(T, size(𝑯,1))
    for i in 1:nnz(𝑯)
        x, y, z, w = 𝑯.index[i]
        𝐯[x] += 6.0 * 𝑯.data[i] * 𝐱[y] * 𝐱[z] * 𝐱[w]
        𝐯[y] += 6.0 * 𝑯.data[i] * 𝐱[x] * 𝐱[z] * 𝐱[w]
        𝐯[z] += 6.0 * 𝑯.data[i] * 𝐱[x] * 𝐱[y] * 𝐱[w]
        𝐯[w] += 6.0 * 𝑯.data[i] * 𝐱[x] * 𝐱[y] * 𝐱[z]
    end
    return 𝐯
end

# handy operator ⊙ (\odot)
⊙ = contract
