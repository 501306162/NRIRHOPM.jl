# unary potentials
function sum_diff_exp{S,T<:Real}(f, fixedImg::AbstractArray, movingImg::AbstractArray, displacements::AbstractArray{SVector{S,T}}, gridDims::NTuple)
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(DimensionMismatch("fixedImg and movingImg must have the same size."))
    length(imageDims) == S || throw(DimensionMismatch("Images and displacement vectors are NOT in the same dimension."))
    cost = zeros(length(displacements), gridDims...)
    # blockDims = imageDims .÷ gridDims
    blockDims = map(div, imageDims, gridDims)
    gridRange = CartesianRange(gridDims)
    𝒊₀ = first(gridRange)
    for a in eachindex(displacements), 𝒊 in gridRange
        # offset = (𝒊 - 𝒊₀).I .* blockDims (pending 0.6)
        offset = map(*, (𝒊 - 𝒊₀).I, blockDims)
        s = zero(Float64)
        for 𝒋 in CartesianRange(blockDims)
            # 𝐤 = offset .+ 𝒋.I
            𝐤 = map(+, offset, 𝒋.I)
            # 𝐝 = 𝐤 .+ blockDims .* displacements[a]
            𝐝 = map(+, 𝐤, map(*, blockDims, displacements[a]))
            if checkbounds(Bool, movingImg, 𝐝...)
                s += e^-f(fixedImg[𝐤...] - movingImg[𝐝...])
            end
        end
        cost[a,𝒊] = s
    end
    return reshape(cost, length(displacements), prod(gridDims))
end

"""
    sadexp(fixedImg, movingImg, displacements)
    sadexp(fixedImg, movingImg, displacements, gridDims)

Calculates the sum of absolute differences between fixed(target) image and
warpped image(moving image + displacements), then applys `f(x)=e⁻ˣ` to the result.
"""
sadexp(fixedImg, movingImg, displacements, gridDims=size(fixedImg)) = sum_diff_exp(abs, fixedImg, movingImg, displacements, gridDims)

"""
    ssdexp(fixedImg, movingImg, displacements)
    ssdexp(fixedImg, movingImg, displacements, gridDims)

Calculates the sum of squared differences between fixed(target) image and
warpped image(moving image + displacements), then applys `f(x)=e⁻ˣ` to the result.
"""
ssdexp(fixedImg, movingImg, displacements, gridDims=size(fixedImg)) = sum_diff_exp(abs2, fixedImg, movingImg, displacements, gridDims)


# pairwise potentials
"""
    potts(fp, fq, d)

Returns the cost value based on Potts model.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43.
"""
@generated function potts{S,Td<:Real}(fp::SVector{S}, fq::SVector{S}, d::Td)
    ex = :(true)
    for i = 1:S
        ex = :($ex && (fp[$i] == fq[$i]))
    end
    return :($ex ? zero(Td) : d)
end

"""
    pottsexp(fp, fq, d)

Calculates the cost value based on Potts model, then applys `f(x)=e⁻ˣ` to the result.
"""
pottsexp(fp, fq, d) = e^-potts(fp, fq, d)


"""
    tad(fp, fq, c, d)

Calculates the truncated absolute difference between `fp` and `fq`.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43-44.
"""
@generated function tad{S,T<:Real}(fp::SVector{S,T}, fq::SVector{S,T}, c::Real, d::Real)
    ex = :(zero(T))
    for i = 1:S
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * sqrt($ex), d))
end

"""
    tadexp(fp, fq, c, d)

Calculates the truncated absolute difference between `fp` and `fq`,
then applys `f(x)=e⁻ˣ` to the result.
"""
tadexp(fp, fq, c, d) = e^-tad(fp, fq, c, d)


"""
    tqd(fp, fq, c, d)

Calculates the truncated quadratic difference between `fp` and `fq`.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 44-45.
"""
@generated function tqd{S,T<:Real}(fp::SVector{S,T}, fq::SVector{S,T}, c::Real, d::Real)
    ex = :(zero(T))
    for i = 1:S
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * $ex, d))
end

"""
    tqdexp(fp, fq, c, d)

Calculates the truncated quadratic difference between `fp` and `fq`,
then applys `f(x)=e⁻ˣ` to the result.
"""
tqdexp(fp, fq, c, d) = e^-tqd(fp, fq, c, d)


"""
    jᶠᶠ(α,β,χ)
    jᵇᶠ(α,β,χ)
    jᶠᵇ(α,β,χ)
    jᵇᵇ(α,β,χ)

Returns the corresponding cost value.

```
coordinate system(r,c):
       +---> c
       |
       ↓
       r
coordinate => point => label:
 𝒊 => p1 => α   𝒋 => p2 => β   𝒌 => p3 => χ
```

Refer to the following paper for further details:

Karacali, Bilge, and Christos Davatzikos. "Estimating topology preserving and
smooth displacement fields." IEEE Transactions on Medical Imaging 23.7 (2004): 870.
"""
jᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+β[1]-α[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(β[2]-α[2])
jᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+α[1]-β[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(α[2]-β[2])
jᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+β[1]-α[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(β[2]-α[2])
jᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+α[1]-β[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(α[2]-β[2])

function jᶠᶠexp(α, β, χ)
    x = jᶠᶠ(α, β, χ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᶠexp(α, β, χ)
    x = jᵇᶠ(α, β, χ)
    return x > 0.0 ? e^-x : 0.0
end
function jᶠᵇexp(α, β, χ)
    x = jᶠᵇ(α, β, χ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᵇexp(α, β, χ)
    x = jᵇᵇ(α, β, χ)
    return x > 0.0 ? e^-x : 0.0
end

"""
    jᶠᶠᶠ(α,β,χ,δ)
    jᵇᶠᶠ(α,β,χ,δ)
    jᶠᵇᶠ(α,β,χ,δ)
    jᵇᵇᶠ(α,β,χ,δ)
    jᶠᶠᵇ(α,β,χ,δ)
    jᵇᶠᵇ(α,β,χ,δ)
    jᶠᵇᵇ(α,β,χ,δ)
    jᵇᵇᵇ(α,β,χ,δ)

Returns the corresponding cost value.

```
coordinate system(r,c,z):
  up  r     c --->        z × × (front to back)
  to  |   left to right     × ×
 down ↓
coordinate => point => label:
 𝒊 => p1 => α   𝒋 => p2 => β   𝒌 => p3 => χ   𝒎 => p5 => δ
```

Refer to the following paper for further details:

Karacali, Bilge, and Christos Davatzikos. "Estimating topology preserving and
smooth displacement fields." IEEE Transactions on Medical Imaging 23.7 (2004): 870.
"""
jᶠᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+β[1]-α[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
  (  δ[1]-α[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
  (  χ[1]-α[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(χ[3]-α[3]))

jᵇᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+α[1]-β[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
  (  δ[1]-α[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
  (  χ[1]-α[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(χ[3]-α[3]))

jᶠᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+β[1]-α[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
  (  δ[1]-α[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
  (  α[1]-χ[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(α[3]-χ[3]))

jᵇᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+α[1]-β[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
  (  δ[1]-α[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
  (  α[1]-χ[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(α[3]-χ[3]))

jᶠᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+β[1]-α[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
  (  α[1]-δ[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
  (  χ[1]-α[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(χ[3]-α[3]))

jᵇᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+α[1]-β[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
  (  α[1]-δ[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
  (  χ[1]-α[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(χ[3]-α[3]))

jᶠᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+β[1]-α[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
  (  α[1]-δ[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
  (  α[1]-χ[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(α[3]-χ[3]))

jᵇᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = (
  (1+α[1]-β[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
  (  α[1]-δ[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
  (  α[1]-χ[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(α[3]-χ[3]))

function jᶠᶠᶠexp(α, β, χ, δ)
    x = jᶠᶠᶠ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᶠᶠexp(α, β, χ, δ)
    x = jᵇᶠᶠ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᶠᵇᶠexp(α, β, χ, δ)
    x = jᶠᵇᶠ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᵇᶠexp(α, β, χ, δ)
    x = jᵇᵇᶠ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᶠᶠᵇexp(α, β, χ, δ)
    x = jᶠᶠᵇ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᶠᵇexp(α, β, χ, δ)
    x = jᵇᶠᵇ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᶠᵇᵇexp(α, β, χ, δ)
    x = jᶠᵇᵇ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end

function jᵇᵇᵇexp(α, β, χ, δ)
    x = jᵇᵇᵇ(α, β, χ, δ)
    return x > 0.0 ? e^-x : 0.0
end
