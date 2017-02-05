"""
    hopm_mixed(𝐡, 𝐇, 𝐒, tol, maxIter, constrainRow) -> (energy, spectrum)
    hopm_mixed(𝐡, 𝐇, 𝑯, 𝐒, tol, maxIter, constrainRow) -> (energy, spectrum)
    hopm_mixed(𝐡, 𝐇, 𝐯, tol, maxIter) -> (energy, vector)
    hopm_mixed(𝐡, 𝐇, 𝑯, 𝐯, tol, maxIter) -> (energy, vector)

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm_mixed{T<:Real}(𝐡::AbstractVector{T}, 𝐇::BSSTensor{T},
                             𝐒::AbstractMatrix{T}, tol::Float64, maxIter::Integer,
                             constrainRow::Bool)
    𝐒₀ = copy(𝐒)
    pixelNum, labelNum = size(𝐒₀)

    if constrainRow
        for r = 1:pixelNum
            normalize!(@view 𝐒₀[r,:])
        end
    else
        𝐒₀ *= 1/vecnorm(𝐒₀)
    end

    𝐌 = reshape(𝐡, pixelNum, labelNum)
    𝐒ᵢ = 𝐒₀
    i = 0
    while i < maxIter
        𝐒ᵢ₊₁ = 𝐌 + 𝐇 ⊙ 𝐒ᵢ
        if constrainRow
            for r = 1:pixelNum
                normalize!(@view 𝐒ᵢ₊₁[r,:])
            end
        else
            𝐒ᵢ₊₁ *= 1/vecnorm(𝐒ᵢ₊₁)
        end
        i += 1
        if vecnorm(𝐒ᵢ₊₁ - 𝐒ᵢ) < tol
            𝐒ᵢ = 𝐒ᵢ₊₁
            break
        end
        𝐒ᵢ = 𝐒ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return sum( 𝐒ᵢ .* (𝐌 + 𝐇 ⊙ 𝐒ᵢ) ), 𝐒ᵢ
end

function hopm_mixed{T<:Real}(𝐡::AbstractVector{T}, 𝐇::BSSTensor{T}, 𝑯::BSSTensor{T},
                             𝐒::AbstractMatrix{T}, tol::Float64, maxIter::Integer,
                             constrainRow::Bool)
    𝐒₀ = copy(𝐒)
    pixelNum, labelNum = size(𝐒₀)

    if constrainRow
        for r = 1:pixelNum
            normalize!(@view 𝐒₀[r,:])
        end
    else
        𝐒₀ *= 1/vecnorm(𝐒₀)
    end

    𝐌 = reshape(𝐡, pixelNum, labelNum)
    𝐒ᵢ = 𝐒₀
    i = 0
    while i < maxIter
        𝐒ᵢ₊₁ = 𝐌 + 𝐇 ⊙ 𝐒ᵢ + 𝑯 ⊙ 𝐒ᵢ
        if constrainRow
            for r = 1:pixelNum
                normalize!(@view 𝐒ᵢ₊₁[r,:])
            end
        else
            𝐒ᵢ₊₁ *= 1/vecnorm(𝐒ᵢ₊₁)
        end
        i += 1
        if vecnorm(𝐒ᵢ₊₁ - 𝐒ᵢ) < tol
            𝐒ᵢ = 𝐒ᵢ₊₁
            break
        end
        𝐒ᵢ = 𝐒ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return sum( 𝐒ᵢ .* (𝐌 + 𝐇 ⊙ 𝐒ᵢ + 𝑯 ⊙ 𝐒ᵢ) ), 𝐒ᵢ
end

function hopm_mixed{T<:Real}(𝐡::AbstractVector{T}, 𝐇::AbstractTensor{T},
                             𝐯::AbstractVector{T}, tol::Float64, maxIter::Integer)
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐡 + 𝐇 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        i += 1
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐡 + 𝐇 ⊙ 𝐯ᵢ), 𝐯ᵢ
end

function hopm_mixed{T<:Real}(𝐡::AbstractVector{T}, 𝐇::AbstractTensor{T}, 𝑯::AbstractTensor{T},
                             𝐯::AbstractVector{T}, tol::Float64, maxIter::Integer)
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐡 + 𝐇 ⊙ 𝐯ᵢ + 𝑯 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        i += 1
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐡 + 𝐇 ⊙ 𝐯ᵢ + 𝑯 ⊙ 𝐯ᵢ), 𝐯ᵢ
end

"""
    hopm_canonical(𝐡, 𝐇, 𝐯, tol, maxIter) -> (energy, vector)
    hopm_canonical(𝐡, 𝐇, 𝑯, 𝐯, tol, maxIter) -> (energy, vector)

The canonical high order power method for calculating tensor eigenpairs.
"""
function hopm_canonical{T<:Real}(𝐡::AbstractVector{T}, 𝐇::AbstractTensor{T},
                                 𝐯::AbstractVector{T}, tol::Float64, maxIter::Integer)
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐡 + 𝐇 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        i += 1
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐡 + 𝐇 ⊙ 𝐯ᵢ), 𝐯ᵢ
end

function hopm_canonical{T<:Real}(𝐡::AbstractVector{T}, 𝐇::AbstractTensor{T}, 𝑯::AbstractTensor{T},
                                 𝐯::AbstractVector{T}, tol::Float64, maxIter::Integer)
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐯ᵢ .* 𝐡 + 𝐯ᵢ .* (𝐇 ⊙ 𝐯ᵢ) + 𝑯 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        i += 1
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐯ᵢ .* 𝐡 + 𝐯ᵢ .* (𝐇 ⊙ 𝐯ᵢ) + 𝑯 ⊙ 𝐯ᵢ), 𝐯ᵢ
end
