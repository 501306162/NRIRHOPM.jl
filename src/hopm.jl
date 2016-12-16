"""
    hopm(𝐇¹, 𝐇²) -> (s, 𝐯)
    hopm(𝐇¹, 𝐇², 𝐇³) -> (s, 𝐯)

The high order power method for calculating tensor eigenpairs.

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, tol::Float64=1e-5, maxIter::Int=100)
    𝐯 = rand(T, length(𝐇¹))
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
    info("HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐇¹ + 𝐇² ⊙ 𝐯ᵢ), 𝐯ᵢ
end

function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, 𝐇³::AbstractTensor{T}, tol::Float64=1e-5, maxIter::Int=100)
    𝐯 = rand(T, length(𝐇¹))
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
    info("HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐇¹ + 𝐇² ⊙ 𝐯ᵢ + 𝐇³ ⊙ 𝐯ᵢ), 𝐯ᵢ
end
