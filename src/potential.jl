# unary potentials
function sum_diff_exp{T,N}(f, fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, displacements::AbstractArray{StaticVector})
    imageDims = indices(fixedImg)
    imageDims == indices(movingImg) || throw(DimensionMismatch("fixedImg and movingImg must have the same indices."))
    movingImgITP = interpolate(movingImg, BSpline(Linear()), OnGrid())
    cost = zeros(length(linearindices(displacements)), length(linearindices(fixedImg)))
    for a in eachindex(displacements), 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        # Todo: 𝐝 = 𝒊.I .+ displacements[a] (pending julia-v0.6)
        𝐝 = map(+, 𝒊.I, displacements[a])
        if Base.checkbounds_indices(Bool, indices(movingImg), 𝐝)
            cost[a,i] = e^-f(fixedImg[𝒊] - movingImgITP[𝐝...])
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
sadexp(fixedImg, movingImg, displacements) = sum_diff_exp(abs, fixedImg, movingImg, displacements)

"""
    ssdexp(fixedImg, movingImg, displacements)

Calculates the sum of squared differences between fixed(target) image and
warpped image(moving image + displacements), then applys `f(x)=e⁻ˣ` to the result.
"""
ssdexp(fixedImg, movingImg, displacements) = sum_diff_exp(abs2, fixedImg, movingImg, displacements)


# pairwise potentials
"""
    potts(fp, fq, d)

Returns the cost value based on Potts model.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43.
"""
@generated function potts{S,T<:Real}(fp::SVector{S,T}, fq::SVector{S,T}, d::T)
    ex = :(true)
    for i = 1:S
        ex = :($ex && (fp[$i] == fq[$i]))
    end
    return :($ex ? zero(T) : d)
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
jᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+β[1]-α[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
jᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+α[1]-β[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0
jᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+β[1]-α[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
jᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}) = (1+α[1]-β[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0

jᶠᶠexp(α, β, χ) = jᶠᶠ(α, β, χ) == 1.0 ? 1.0 : e^-1
jᵇᶠexp(α, β, χ) = jᵇᶠ(α, β, χ) == 1.0 ? 1.0 : e^-1
jᶠᵇexp(α, β, χ) = jᶠᵇ(α, β, χ) == 1.0 ? 1.0 : e^-1
jᵇᵇexp(α, β, χ) = jᵇᵇ(α, β, χ) == 1.0 ? 1.0 : e^-1

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
jᶠᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
                                                                                      (   δ[1]-α[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                                      (   χ[1]-α[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᵇᶠᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (  χ[1]-α[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
                                                                                      (   δ[1]-α[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  δ[1]-α[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                                      (   χ[1]-α[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᶠᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(β[3]-α[3]) +
                                                                                      (   δ[1]-α[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                                      (   α[1]-χ[1])*(  β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(  δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᵇᵇᶠ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (  α[1]-χ[1])*(  δ[2]-α[2])*(α[3]-β[3]) +
                                                                                      (   δ[1]-α[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  δ[1]-α[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                                      (   α[1]-χ[1])*(  α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(  δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᶠᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
                                                                                      (   α[1]-δ[1])*(  β[2]-α[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                                      (   χ[1]-α[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᵇᶠᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (  χ[1]-α[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
                                                                                      (   α[1]-δ[1])*(  α[2]-β[2])*(  χ[3]-α[3]) - (  α[1]-δ[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                                      (   χ[1]-α[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᶠᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(β[3]-α[3]) +
                                                                                      (   α[1]-δ[1])*(  β[2]-α[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                                      (   α[1]-χ[1])*(  β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(  α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᵇᵇᵇ{S,T<:Real}(α::SVector{S,T}, β::SVector{S,T}, χ::SVector{S,T}, δ::SVector{S,T}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (  α[1]-χ[1])*(  α[2]-δ[2])*(α[3]-β[3]) +
                                                                                      (   α[1]-δ[1])*(  α[2]-β[2])*(  α[3]-χ[3]) - (  α[1]-δ[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                                      (   α[1]-χ[1])*(  α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(  α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᶠᶠᶠexp(α, β, χ, δ) = jᶠᶠᶠ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᵇᶠᶠexp(α, β, χ, δ) = jᵇᶠᶠ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᶠᵇᶠexp(α, β, χ, δ) = jᶠᵇᶠ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᵇᵇᶠexp(α, β, χ, δ) = jᵇᵇᶠ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᶠᶠᵇexp(α, β, χ, δ) = jᶠᶠᵇ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᵇᶠᵇexp(α, β, χ, δ) = jᵇᶠᵇ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᶠᵇᵇexp(α, β, χ, δ) = jᶠᵇᵇ(α, β, χ, δ) == 1 ? 1.0 : e^-1
jᵇᵇᵇexp(α, β, χ, δ) = jᵇᵇᵇ(α, β, χ, δ) == 1 ? 1.0 : e^-1
