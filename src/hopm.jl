using TensorOperations, TensorDecompositions


"""
high order (mixed) power method
inspired by https://github.com/yunjhongwu/TensorDecompositions.jl
"""
function hopm{T}(tensor::AbstractArray{T};
                 tol::Float64=1e-5,
                 maxiter::Int=1000
                )

    x = randn(size(tensor, 1))
    x .*= 1/vecnorm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        copy!(x_old, x)
        x = A_mul_B(tensor, x)
        x *= 1/vecnorm(x)
        converged = vecnorm(x - x_old) < tol
        niters += 1
    end
    return dot(x, A_mul_B(tensor, x)), x
end

# first order & second order mixed hopm
function hopm{T}(tensor¹::AbstractArray{T,1},
                 tensor²::AbstractArray{T,2},
                 λ::Float64;
                 tol::Float64=1e-5,
                 maxiter::Int=1000
                )

    r = size(tensor¹, 1)
    r != size(tensor², 1) && error("checkout tensors")
    x = randn(r)
    x .*= 1/vecnorm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        copy!(x_old, x)
        x = A_mul_B(tensor¹, x) + λ * A_mul_B(tensor², x)
        x *= 1/vecnorm(x)
        converged = vecnorm(x - x_old) < tol
        niters += 1
		@show niters
    end
    @show niters
    return dot(x, A_mul_B(tensor¹, x) + λ * A_mul_B(tensor², x)), x
end

# 123 mixed hopm
function hopm{T}(tensor¹::AbstractArray{T,1},
                 tensor²::AbstractArray{T,2},
                 tensor³::AbstractArray{T,3},
                 λ::Float64,
                 Θ::Float64;
                 tol::Float64=1e-5,
                 maxiter::Int=1000
                )

    r = size(tensor¹, 1)
    r != size(tensor², 1) && error("checkout tensors --2")
    r != size(tensor³, 1) && error("checkout tensors --3")
    x = randn(r)
    x .*= 1/vecnorm(x)
    x_old = similar(x)
    converged = false
    niters = 0
    while !converged && niters < maxiter
        copy!(x_old, x)
        x = A_mul_B(tensor¹, x) + λ*A_mul_B(tensor², x) + Θ*A_mul_B(tensor³, x)
        x *= 1/vecnorm(x)
        converged = vecnorm(x - x_old) < tol
        niters += 1
		@show niters
    end
    @show niters
    return dot(x, A_mul_B(tensor¹, x) + λ*A_mul_B(tensor², x) + Θ*A_mul_B(tensor³, x)), x
end

# Tensorcontract
function A_mul_B{T,N}(tensor::StridedArray{T,N}, x::Vector{T})
    v = copy(tensor)
    for i in 2:N
        v = tensorcontract(v, collect(i-1:N), x, N)
    end
    return v
end

function A_mul_B{T,N}(tnsr::SparseArray{T,N}, x::Vector{T})
    v = zeros(T, size(tnsr, 1))
    for i in 1:nnz(tnsr)
        v[tnsr.pos[1, i]] += tnsr.vals[i] * prod(x[tnsr.pos[2:N, i]])
    end
    return v
end
