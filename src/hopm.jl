function constrain!(x::AbstractMatrix, constraint::Symbol)
    if constraint == :vecnorm
        x .*= 1/vecnorm(x)
    elseif constraint == :column
        for c in indices(x,2)
            normalize!(@view x[:,c])
        end
    end
    nothing
end

"""
    hopm_mixed(𝐭, 𝐓, 𝐌, constraint, tol, maxIter) -> (energy, spectrum)
    hopm_mixed(𝐭, 𝐓, 𝑻, 𝐌, constraint, tol, maxIter) -> (energy, spectrum)

Refer to the following paper(Algorithm 4) for further details:

Duchenne, Olivier, et al. "A tensor-based algorithm for high-order graph matching."
IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2383-2395.
"""
function hopm_mixed(𝐭::AbstractMatrix, 𝐓::BlockedTensor, 𝐌::AbstractMatrix,
                    constraint::Symbol, tol::Float64, maxIter::Integer)
    𝐌₀ = copy(𝐌)
    constrain!(𝐌₀, constraint)
    𝐌ᵢ = 𝐌₀
    i = 0
    while i < maxIter
        i += 1
        𝐌ᵢ₊₁ = 𝐭 + 𝐓 ⊙ 𝐌ᵢ
        constrain!(𝐌ᵢ₊₁, constraint)
        if vecnorm(𝐌ᵢ₊₁ - 𝐌ᵢ) < tol
            𝐌ᵢ = 𝐌ᵢ₊₁
            break
        end
        𝐌ᵢ = 𝐌ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return sum(𝐌ᵢ .* (𝐭 + 𝐓 ⊙ 𝐌ᵢ)), 𝐌ᵢ
end

function hopm_mixed(𝐭::AbstractMatrix, 𝐓::BlockedTensor, 𝑻::BlockedTensor, 𝐌::AbstractMatrix,
                    constraint::Symbol, tol::Float64, maxIter::Integer)
    𝐌₀ = copy(𝐌)
    constrain!(𝐌₀, constraint)
    𝐌ᵢ = 𝐌₀
    i = 0
    while i < maxIter
        i += 1
        𝐌ᵢ₊₁ = 𝐭 + 𝐓 ⊙ 𝐌ᵢ + 𝑻 ⊙ 𝐌ᵢ
        constrain!(𝐌ᵢ₊₁, constraint)
        if vecnorm(𝐌ᵢ₊₁ - 𝐌ᵢ) < tol
            𝐌ᵢ = 𝐌ᵢ₊₁
            break
        end
        𝐌ᵢ = 𝐌ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, HOPM might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return sum(𝐌ᵢ .* (𝐭 + 𝐓 ⊙ 𝐌ᵢ + 𝑻 ⊙ 𝐌ᵢ)), 𝐌ᵢ
end


"""
    hopm_canonical(𝐭, 𝐓, 𝐌, tol, maxIter) -> (energy, spectrum)
    hopm_canonical(𝐭, 𝐓, 𝑻, 𝐌, tol, maxIter) -> (energy, spectrum)

The canonical high order power method for calculating tensor eigenpairs.
"""
function hopm_canonical(𝐭::AbstractMatrix, 𝐓::BlockedTensor, 𝐌::AbstractMatrix,
                        tol::Float64, maxIter::Integer
                       )
    e, 𝐯 = hopm_canonical(reshape(𝐭, length(𝐭)), 𝐓, reshape(𝐌, length(𝐌)), tol, maxIter)
    return e, reshape(𝐯, size(𝐭))
end

function hopm_canonical(𝐭::AbstractMatrix, 𝐓::BlockedTensor, 𝑻::BlockedTensor, 𝐌::AbstractMatrix,
                        tol::Float64, maxIter::Integer
                       )
    e, 𝐯 = hopm_canonical(reshape(𝐭, length(𝐭)), 𝐓, 𝑻, reshape(𝐌, length(𝐌)), tol, maxIter)
    return e, reshape(𝐯, size(𝐭))
end

function hopm_canonical(𝐭::AbstractVector, 𝐓::BlockedTensor, 𝐯::AbstractVector,
                        tol::Float64, maxIter::Integer
                       )
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        i += 1
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐭 + 𝐓 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, the result might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐭 + 𝐓 ⊙ 𝐯ᵢ), 𝐯ᵢ
end

function hopm_canonical(𝐭::AbstractVector, 𝐓::BlockedTensor, 𝑻::BlockedTensor, 𝐯::AbstractVector,
                        tol::Float64, maxIter::Integer
                       )
    𝐯₀ = copy(𝐯)
    normalize!(𝐯₀)
    𝐯ᵢ = 𝐯₀
    i = 0
    while i < maxIter
        i += 1
        𝐯ᵢ₊₁ = 𝐯ᵢ .* 𝐯ᵢ .* 𝐭 + 𝐯ᵢ .* (𝐓 ⊙ 𝐯ᵢ) + 𝑻 ⊙ 𝐯ᵢ
        normalize!(𝐯ᵢ₊₁)
        if vecnorm(𝐯ᵢ₊₁ - 𝐯ᵢ) < tol
            𝐯ᵢ = 𝐯ᵢ₊₁
            break
        end
        𝐯ᵢ = 𝐯ᵢ₊₁
    end
    logger = get_logger(current_module())
    i == maxIter && warn(logger, "Maximum iterator number is reached, the result might not be convergent.")
    i < maxIter && info(logger, "HOPM converges in $i steps.")
    return 𝐯ᵢ ⋅ (𝐯ᵢ .* 𝐯ᵢ .* 𝐭 + 𝐯ᵢ .* (𝐓 ⊙ 𝐯ᵢ) + 𝑻 ⊙ 𝐯ᵢ), 𝐯ᵢ
end
