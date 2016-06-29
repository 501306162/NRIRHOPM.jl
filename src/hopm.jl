# The following code is mainly inspired by https://github.com/yunjhongwu/TensorDecompositions.jl

immutable SparseArray{T, N} <: AbstractArray{T, N}
    vals::Vector{T}
    pos::Matrix{Int}
    dims::NTuple{N,Int}
end

"""
SharedSparseTensor
"""
immutable SharedSparseTensor{T, N} <: AbstractArray{T, N}
    values::SharedArray{T, 1}
    indices::SharedArray{Int, 2}
    dims::NTuple{N, Int}
end

Base.size(A::SharedSparseTensor) = A.dims
Base.size(A::SharedSparseTensor, i::Int) = A.dims[i]
Base.nnz(A::SharedSparseTensor) = length(A.values)
Base.length(A::SharedSparseTensor) = prod(A.dims)

function share{T}(A::AbstractArray{T})
    sh = SharedArray(T, size(A))
    for i=1:length(A)
        sh.s[i] = A[i]
    end
    return sh
end

"""
High Order (Mixed) Power Method
"""
function hopm{T}(tensor¹::AbstractArray{T,1},
                 tensor²::AbstractArray{T,2},
                 λ::Float64;
                 tol::Float64=1e-5,
                 maxiter::Int=1000
                )
    r = size(tensor¹, 1)
    r != size(tensor², 1) && error("Tensor Dimension Mismatch.")
    x = randn(r)
    x .*= 1/vecnorm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        x_old = deepcopy(x)
        x = tensor¹ + λ * A_mul_B(tensor², share(x))
        x *= 1/vecnorm(x)
        converged = vecnorm(x - x_old) < tol
        niters += 1
		@show niters
    end
    @show niters
    return dot(x, tensor¹ + λ * A_mul_B(tensor², share(x))), x
end

"""
Tensor Contraction
"""
function A_mul_B{T,N}(tensor::SharedSparseTensor{T,N}, x::SharedArray{T,1})
	v = SharedArray(T, size(tensor, 1))
    @sync @parallel for i in 1:nnz(tensor)
        v[tensor.indices[1, i]] += tensor.values[i] * prod(x[tensor.indices[2:N, i]])
    end
    return sdata(v)
end
