# unary potentials
@generated function sum_diff_exp{N,Ti<:Real,Td<:Real}(f, fixedImg::AbstractArray{Ti,N}, movingImg::AbstractArray{Ti,N}, displacements::AbstractArray{SVector{N,Td}}, gridDims::NTuple)
    quote
        imageDims = size(fixedImg)
        imageDims == size(movingImg) || throw(DimensionMismatch("fixedImg and movingImg must have the same size."))
        length(imageDims) == $N || throw(DimensionMismatch("Images and displacement vectors are NOT in the same dimension."))
        # blockDims = imageDims .÷ gridDims
        blockDims = map(div, imageDims, gridDims)
        cost = zeros(length(displacements), gridDims...)
        for a in eachindex(displacements), i in CartesianRange(gridDims)
            @nexprs $N x->offset_x = (i[x] - 1) * blockDims[x]
            s = zero(Float64)
            for j in CartesianRange(blockDims)
                @nexprs $N x->k_x = offset_x + j[x]
                @nexprs $N x->d_x = k_x + blockDims[x] * displacements[a][x]
                if @nall $N x->(1 ≤ d_x ≤ imageDims[x])
                    fixed = @nref $N fixedImg k
                    moving = @nref $N movingImg d
                    s += e^-f(fixed - moving)
                end
            end
            cost[a,i] = s
        end
        reshape(cost, length(displacements), prod(gridDims))
    end
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
@generated potts{S,Td<:Real}(fp::SVector{S}, fq::SVector{S}, d::Td) = :((@nall $S x->(fp[x] == fq[x])) ? zero(Td) : d)

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


# high-order potentials
"""
    p3potts(fp, fq, fr, d)

Returns the cost value based on P³ Potts model.

Refer to the following paper for further details:

Kohli, Pushmeet, M. Pawan Kumar, and Philip HS Torr. "Solving energies with higher order cliques." In CVPR. 2007.
"""
@generated p3potts{S,Td<:Real}(fp::SVector{S}, fq::SVector{S}, fr::SVector{S}, d::Td) = :((@nall $S x->(fp[x] == fq[x] == fr[x])) ? zero(Td) : d)

"""
    p3pottsexp(fp, fq, fr, d)

Calculates the cost value based on Potts model, then applys `f(x)=e⁻ˣ` to the result.
"""
p3pottsexp(fp, fq, fr, d) = e^-p3potts(fp, fq, fr, d)

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

jᶠᶠexp(α, β, χ) = jᶠᶠ(α, β, χ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᶠexp(α, β, χ) = jᵇᶠ(α, β, χ) |> x->x > 0.0 ? e^-x : 0.0
jᶠᵇexp(α, β, χ) = jᶠᵇ(α, β, χ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᵇexp(α, β, χ) = jᵇᵇ(α, β, χ) |> x->x > 0.0 ? e^-x : 0.0

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

jᶠᶠᶠexp(α, β, χ, δ) = jᶠᶠᶠ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᶠᶠexp(α, β, χ, δ) = jᵇᶠᶠ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᶠᵇᶠexp(α, β, χ, δ) = jᶠᵇᶠ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᵇᶠexp(α, β, χ, δ) = jᵇᵇᶠ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᶠᶠᵇexp(α, β, χ, δ) = jᶠᶠᵇ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᶠᵇexp(α, β, χ, δ) = jᵇᶠᵇ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᶠᵇᵇexp(α, β, χ, δ) = jᶠᵇᵇ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
jᵇᵇᵇexp(α, β, χ, δ) = jᵇᵇᵇ(α, β, χ, δ) |> x->x > 0.0 ? e^-x : 0.0
