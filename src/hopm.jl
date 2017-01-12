"""
    hopm(𝐇¹, 𝐇², 𝐯) -> (e, 𝐯)
    hopm(𝐇¹, 𝐇², 𝐒) -> (e, 𝐒)
    hopm(𝐇¹, 𝐇², 𝐇³⁺, 𝐯) -> (e, 𝐯)
    hopm(𝐇¹, 𝐇², 𝐇³⁺, 𝐒) -> (e, 𝐒)

The high order power method for calculating tensor eigenpairs.

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, 𝐒::AbstractMatrix{T},
                       tol::Float64=1e-5, maxIter::Int=300, verbose::Bool=false)
    𝐒₀ = copy(𝐒)
    pixelNum, labelNum = size(𝐒₀)
    𝐌¹ = reshape(𝐇¹, pixelNum, labelNum)
    # only constrain rows
    for i = 1:pixelNum
        normalize!(@view 𝐒₀[i,:])
    end
    𝐒ᵢ = 𝐒₀
    i = 0
    while i < maxIter
        𝐒ᵢ₊₁ = 𝐌¹ + 𝐇² ⊙ 𝐒ᵢ
        # only constrain rows
        for i = 1:pixelNum
            normalize!(@view 𝐒ᵢ₊₁[i,:])
        end
        vecnorm(𝐒ᵢ₊₁ - 𝐒ᵢ) < tol && break
        i += 1
        𝐒ᵢ = 𝐒ᵢ₊₁
    end
    if i == maxIter
        warn("Maximum iterator number is reached, HOPM could not be convergent.")
    else
        verbose && info("HOPM converges in $i steps.")
    end
    return sum(𝐒ᵢ.*(𝐌¹ + 𝐇² ⊙ 𝐒ᵢ)), 𝐒ᵢ
end

function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, 𝐇³⁺::AbstractTensor{T}, 𝐒::AbstractMatrix{T},
                       tol::Float64=1e-5, maxIter::Int=300, verbose::Bool=false)
    𝐒₀ = copy(𝐒)
    pixelNum, labelNum = size(𝐒₀)
    𝐌¹ = reshape(𝐇¹, pixelNum, labelNum)
    # only constrain rows
    for i = 1:pixelNum
        normalize!(@view 𝐒₀[i,:])
    end
    𝐒ᵢ = 𝐒₀
    i = 0
    while i < maxIter
        𝐒ᵢ₊₁ = 𝐌¹ + 𝐇² ⊙ 𝐒ᵢ + 𝐇³⁺ ⊙ 𝐒ᵢ
        # only constrain rows
        for i = 1:pixelNum
            normalize!(@view 𝐒ᵢ₊₁[i,:])
        end
        vecnorm(𝐒ᵢ₊₁ - 𝐒ᵢ) < tol && break
        i += 1
        𝐒ᵢ = 𝐒ᵢ₊₁
    end
    if i == maxIter
        warn("Maximum iterator number is reached, HOPM could not be convergent.")
    else
        verbose && info("HOPM converges in $i steps.")
    end
    return sum(𝐒ᵢ.*(𝐌¹ + 𝐇² ⊙ 𝐒ᵢ + 𝐇³⁺ ⊙ 𝐒ᵢ)), 𝐒ᵢ
end


function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, 𝐯::AbstractVector{T},
                       tol::Float64=1e-5, maxIter::Int=300, verbose::Bool=false)
    𝐯₀ = 𝐯/vecnorm(𝐯)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐇¹ + 𝐇² ⊙ 𝐯ᵢ
        𝐯ᵢ₊₁ = 𝐯ᵢ₊₁/vecnorm(𝐯ᵢ₊₁)
        vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol && break
        i += 1
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    if i == maxIter
        warn("Maximum iterator number is reached, HOPM could not be convergent.")
    else
        verbose && info("HOPM converges in $i steps.")
    end
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐇¹ + 𝐇² ⊙ 𝐯ᵢ), 𝐯ᵢ
end

function hopm{T<:Real}(𝐇¹::AbstractArray{T,1}, 𝐇²::AbstractTensor{T}, 𝐇³⁺::AbstractTensor{T}, 𝐯::AbstractVector{T},
                       tol::Float64=1e-5, maxIter::Int=300, verbose::Bool=false)
    𝐯₀ = 𝐯/vecnorm(𝐯)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐯ᵢ .* 𝐇¹ + 𝐯ᵢ .* (𝐇² ⊙ 𝐯ᵢ) + 𝐇³⁺ ⊙ 𝐯ᵢ
        𝐯ᵢ₊₁ = 𝐯ᵢ₊₁/vecnorm(𝐯ᵢ₊₁)
        vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol && break
        i += 1
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    if i == maxIter
        warn("Maximum iterator number is reached, HOPM could not be convergent.")
    else
        verbose && info("HOPM converges in $i steps.")
    end
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐯ᵢ .* 𝐇¹ + 𝐯ᵢ .* (𝐇² ⊙ 𝐯ᵢ) + 𝐇³⁺ ⊙ 𝐯ᵢ), 𝐯ᵢ
end
