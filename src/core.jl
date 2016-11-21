"""
"Pure" Sparse Symmetric Tensor
"""
immutable PSSTensor{Tv<:Real,Ti<:Integer,Order} <: AbstractArray{Tv, Order}
    data::Vector{Tv}
    index::Vector{NTuple{Order,Ti}}
    dims::NTuple{Order,Ti}
end

Base.nnz(A::PSSTensor) = length(A.data)
Base.size(A::PSSTensor) = A.dims
Base.size(A::PSSTensor, i::Integer) = A.dims[i]
Base.length(A::PSSTensor) = prod(A.dims)

function pcontract{Tv<:Real,Ti<:Integer}(𝐇::PSSTensor{Tv,Ti,2}, 𝐱::Vector{Tv})
    𝐯 = zeros(Tv, size(𝐇,1))
    @inbounds for i in 1:nnz(𝐇)
        x, y = 𝐇.index[i]
        value = 𝐇.data[i]
        𝐯[x] += value * 𝐱[y]
        𝐯[y] += value * 𝐱[x]
    end
    return 𝐯
end

function pcontract{Tv<:Real,Ti<:Integer}(𝐇::PSSTensor{Tv,Ti,3}, 𝐱::Vector{Tv})
    𝐯 = zeros(Tv, size(𝐇,1))
    @inbounds for i in 1:nnz(𝐇)
        x, y, z = 𝐇.index[i]
        value = 𝐇.data[i]
        𝐯[x] += 2.0 * value * 𝐱[y] * 𝐱[z]
        𝐯[y] += 2.0 * value * 𝐱[x] * 𝐱[z]
        𝐯[z] += 2.0 * value * 𝐱[x] * 𝐱[y]
    end
    return 𝐯
end

⊙ = pcontract

"""
    hopm(𝐇¹, 𝐇²) -> (s, 𝐯)

The high order power method for first and second order tensor.

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm{Tv,Ti}(
    𝐇¹::AbstractArray{Tv,1},
    𝐇²::PSSTensor{Tv,Ti,2},
    tol::Float64=1e-5,
    maxIter::Int=100
    )
    size(𝐇¹, 1) != size(𝐇², 1) && throw(ArgumentError("Tensor Dimension Mismatch!"))
    𝐯 = rand(Tv, length(𝐇¹))
    𝐯₀ = 𝐯/vecnorm(𝐯)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐇¹ + 𝐇² ⊙ 𝐯ᵢ
        𝐯ᵢ₊₁ = 𝐯ᵢ₊₁/vecnorm(𝐯ᵢ₊₁)
        vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol && break
        i += 1
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    @show i
    return 𝐯ᵢ ⋅ (𝐇¹ + 𝐇² ⊙ 𝐯ᵢ), 𝐯ᵢ
end

"""
    hopm(𝐇¹, 𝐇², 𝐇³) -> (s, 𝐯)

The high order power method for first, second and third order tensor.

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm{Tv,Ti}(
    𝐇¹::AbstractArray{Tv,1},
    𝐇²::PSSTensor{Tv,Ti,2},
    𝐇³::PSSTensor{Tv,Ti,3},
    tol::Float64=1e-5,
    maxIter::Int=100
    )
    size(𝐇¹, 1) != size(𝐇², 1) && throw(ArgumentError("Tensor Dimension Mismatch!"))
    size(𝐇¹, 1) != size(𝐇³, 1) && throw(ArgumentError("Tensor Dimension Mismatch!"))
    𝐯 = rand(Tv, length(𝐇¹))
    𝐯₀ = 𝐯/vecnorm(𝐯)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐇¹ + 𝐇² ⊙ 𝐯ᵢ + 𝐇³ ⊙ 𝐯ᵢ
        𝐯ᵢ₊₁ = 𝐯ᵢ₊₁/vecnorm(𝐯ᵢ₊₁)
        vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol && break
        i += 1
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    @show i
    return 𝐯ᵢ ⋅ (𝐇¹ + 𝐇² ⊙ 𝐯ᵢ + 𝐇³ ⊙ 𝐯ᵢ), 𝐯ᵢ
end
