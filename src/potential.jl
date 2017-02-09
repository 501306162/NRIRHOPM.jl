# unary potentials
@inline function sum_diff_exp{T,N}(f, fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, displacements::AbstractArray{NTuple{N}})
    imageDims = indices(fixedImg)
    imageDims == indices(movingImg) || throw(DimensionMismatch("fixedImg and movingImg must have the same indices."))
    cost = zeros(length(linearindices(displacements)), length(linearindices(fixedImg)))
    for a in eachindex(displacements), 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        𝒅 = 𝒊 + CartesianIndex(displacements[a])
        if checkbounds(Bool, movingImg, 𝒅)
            cost[a,i] = e^-f(fixedImg[𝒊] - movingImg[𝒅])
        else
            cost[a,i] = 0
        end
    end
    return cost
end

"""
    sadexp(fixedImg, movingImg, displacements)

Calculates the sum of absolute differences between fixed(target) image and
warpped image(moving image + displacements), then applys `f(x)=e⁻ˣ` to the result.
"""
@inline sadexp(fixedImg, movingImg, displacements) = sum_diff_exp(abs, fixedImg, movingImg, displacements)

"""
    ssdexp(fixedImg, movingImg, displacements)

Calculates the sum of squared differences between fixed(target) image and
warpped image(moving image + displacements), then applys `f(x)=e⁻ˣ` to the result.
"""
@inline ssdexp(fixedImg, movingImg, displacements) = sum_diff_exp(abs2, fixedImg, movingImg, displacements)


# pairwise potentials
"""
    potts(fp, fq, d)

Returns the cost value based on Potts model.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43.
"""
@generated function potts{T<:Real,N}(fp::NTuple{N}, fq::NTuple{N}, d::T)
    ex = :(true)
    for i = 1:N
        ex = :($ex && (fp[$i] == fq[$i]))
    end
    return :($ex ? zero(T) : d)
end

"""
    pottsexp(fp, fq, d)

Calculates the cost value based on Potts model, then applys `f(x)=e⁻ˣ` to the result.
"""
@inline pottsexp(fp, fq, d) = e^-potts(fp, fq, d)


"""
    tad(fp, fq, c, d)

Calculates the truncated absolute difference between `fp` and `fq`.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43-44.
"""
@generated function tad{N,T<:Real}(fp::NTuple{N,T}, fq::NTuple{N,T}, c::Real, d::Real)
    ex = :(zero(T))
    for i = 1:N
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * sqrt($ex), d))
end

"""
    tadexp(fp, fq, c, d)

Calculates the truncated absolute difference between `fp` and `fq`,
then applys `f(x)=e⁻ˣ` to the result.
"""
@inline tadexp(fp, fq, c, d) = e^-tad(fp, fq, c, d)


"""
    tqd(fp, fq, c, d)

Calculates the truncated quadratic difference between `fp` and `fq`.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 44-45.
"""
@generated function tqd{N,T<:Real}(fp::NTuple{N,T}, fq::NTuple{N,T}, c::Real, d::Real)
    ex = :(zero(T))
    for i = 1:N
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * $ex, d))
end

"""
    tqdexp(fp, fq, c, d)

Calculates the truncated quadratic difference between `fp` and `fq`,
then applys `f(x)=e⁻ˣ` to the result.
"""
@inline tqdexp(fp, fq, c, d) = e^-tqd(fp, fq, c, d)


"""
    jᶠᶠ(α,β,χ)
    jᵇᶠ(α,β,χ)
    jᶠᵇ(α,β,χ)
    jᵇᵇ(α,β,χ)

Returns the corresponding cost value: 1 => topology preserving, 0 => otherwise.

```
coordinate system(r,c):
       +---> c
       |
       ↓
       r
coordinate => point => label:
 ii => p1 => α   jj => p2 => β   kk => p3 => χ
```

Refer to the following paper for further details:

Karacali, Bilge, and Christos Davatzikos. "Estimating topology preserving and
smooth displacement fields." IEEE Transactions on Medical Imaging 23.7 (2004): 870.
"""
@inline jᶠᶠ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+β[1]-α[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
@inline jᵇᶠ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+α[1]-β[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0
@inline jᶠᵇ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+β[1]-α[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
@inline jᵇᵇ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+α[1]-β[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0


"""
    jᶠᶠᶠ(α,β,χ,δ)
    jᵇᶠᶠ(α,β,χ,δ)
    jᶠᵇᶠ(α,β,χ,δ)
    jᵇᵇᶠ(α,β,χ,δ)
    jᶠᶠᵇ(α,β,χ,δ)
    jᵇᶠᵇ(α,β,χ,δ)
    jᶠᵇᵇ(α,β,χ,δ)
    jᵇᵇᵇ(α,β,χ,δ)

Returns the corresponding cost value: 1 => topology preserving, 0 => otherwise.

```
coordinate system(r,c,z):
  up  r     c --->        z × × (front to back)
  to  |   left to right     × ×
 down ↓
coordinate => point => label:
 iii => p1 => α   jjj => p2 => β   kkk => p3 => χ   mmm => p5 => δ
```

Refer to the following paper for further details:

Karacali, Bilge, and Christos Davatzikos. "Estimating topology preserving and
smooth displacement fields." IEEE Transactions on Medical Imaging 23.7 (2004): 870.
"""
@inline jᶠᶠᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
                                                                       (   δ[1]-α[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                       (   χ[1]-α[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

@inline jᵇᶠᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
                                                                       (   δ[1]-α[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                       (   χ[1]-α[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

@inline jᶠᵇᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
                                                                       (   δ[1]-α[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                       (   α[1]-χ[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

@inline jᵇᵇᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
                                                                       (   δ[1]-α[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                       (   α[1]-χ[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

@inline jᶠᶠᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
                                                                       (   α[1]-δ[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                       (   χ[1]-α[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

@inline jᵇᶠᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
                                                                       (   α[1]-δ[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                       (   χ[1]-α[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

@inline jᶠᵇᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
                                                                       (   α[1]-δ[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                       (   α[1]-χ[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

@inline jᵇᵇᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
                                                                       (   α[1]-δ[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                       (   α[1]-χ[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0
