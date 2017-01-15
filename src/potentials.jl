# unary potentials
"""
    sum_absolute_diff(fixedImg, movingImg, labels)

Calculates the sum of absolute differences between fixed(target) image
and moving(source) image.
"""
function sum_absolute_diff{T,N}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}})
    imageDims = size(fixedImg)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    cost = zeros(pixelNum, labelNum)
    for 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        for a in eachindex(labels)
            𝐝 = 𝒊 + CartesianIndex(labels[a])
            if checkbounds(Bool, movingImg, 𝐝)
                cost[i,a] = e^-abs(fixedImg[𝒊] - movingImg[𝐝])
            else
                cost[i,a] = 0
            end
        end
    end
    return reshape(cost, pixelNum * labelNum)
end

"""
    sum_squared_diff(fixedImg, movingImg, labels)

Calculates the sum of squared differences between fixed(target) image
and moving(source) image.
"""
function sum_squared_diff{T,N}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}})
    imageDims = size(fixedImg)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    cost = zeros(pixelNum, labelNum)
    for 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        for a in eachindex(labels)
            𝐝 = 𝒊 + CartesianIndex(labels[a])
            if checkbounds(Bool, movingImg, 𝐝)
                cost[i,a] = e^-abs2(fixedImg[𝒊] - movingImg[𝐝])
            else
                cost[i,a] = 0
            end
        end
    end
    return reshape(cost, pixelNum * labelNum)
end


# pairwise potentials
"""
    potts_model(fp, fq, d)

Returns the cost value based on Potts model.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43.
"""
@generated function potts_model{T<:Real,N}(fp::NTuple{N}, fq::NTuple{N}, d::T)
    ex = :(true)
    for i = 1:N
        ex = :($ex && (fp[$i] == fq[$i]))
    end
    return :($ex ? T(0) : d)
end


"""
    truncated_absolute_diff(fp, fq, c, d)

Calculates the truncated absolute difference between two labels.
Returns the cost value.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 43-44.
"""
@generated function truncated_absolute_diff{N}(fp::NTuple{N}, fq::NTuple{N}, c::Real, d::Real)
    ex = :(0)
    for i = 1:N
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * sqrt($ex), d))
end


"""
    truncated_quadratic_diff(fp, fq, c, d)

Calculates the truncated quadratic difference between two labels.
Returns the cost value.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 44-45.
"""
@generated function truncated_quadratic_diff{N}(fp::NTuple{N}, fq::NTuple{N}, c::Real, d::Real)
    ex = :(0)
    for i = 1:N
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * $ex, d))
end


"""
    topology_preserving(s₁, s₂, s₃, a, b, c)

Returns the cost value: 1 => topology preserving, 0 => otherwise.
Note that the coordinate system is:

```
       y
       ↑
       |
(x,y): +---> x
```

Refer to the following paper for further details:

Cordero-Grande, Lucilio, et al. "A Markov random field approach for
topology-preserving registration: Application to object-based tomographic image
interpolation." IEEE Transactions on Image Processing 21.4 (2012): 2051.
"""
@inline function topology_preserving{T<:Integer}(s₁::Vector{T}, s₂::Vector{T}, s₃::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T})
    @inbounds begin
        𝐤s₁, 𝐤s₂, 𝐤s₃ = s₁ + a, s₂ + b, s₃ + c
        ∂φ₁∂φ₂ = (𝐤s₂[1] - 𝐤s₁[1]) * (𝐤s₂[2] - 𝐤s₃[2])
        ∂φ₂∂φ₁ = (𝐤s₂[2] - 𝐤s₁[2]) * (𝐤s₂[1] - 𝐤s₃[1])
        ∂r₁∂r₂ = (s₂[1] - s₁[1])*(s₂[2] - s₃[2])
    end
    v = (∂φ₁∂φ₂ - ∂φ₂∂φ₁) / ∂r₁∂r₂
    return v > 0 ? 1.0 : 0.0
end

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
jᶠᶠ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+β[1]-α[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
jᵇᶠ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+α[1]-β[1])*(1+χ[2]-α[2]) - (χ[1]-α[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0
jᶠᵇ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+β[1]-α[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(β[2]-α[2]) > 0 ? 1.0 : 0.0
jᵇᵇ{N}(α::NTuple{N}, β::NTuple{N}, χ::NTuple{N}) = (1+α[1]-β[1])*(1+α[2]-χ[2]) - (α[1]-χ[1])*(α[2]-β[2]) > 0 ? 1.0 : 0.0

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
jᶠᶠᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (χ[1]-α[1])*(δ[2]-α[2])*(β[3]-α[3]) +
                                                                (δ[1]-α[1])*(β[2]-α[2])*(χ[3]-α[3]) - (δ[1]-α[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                (χ[1]-α[1])*(β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᵇᶠᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+δ[3]-α[3]) + (χ[1]-α[1])*(δ[2]-α[2])*(α[3]-β[3]) +
                                                                (δ[1]-α[1])*(α[2]-β[2])*(χ[3]-α[3]) - (δ[1]-α[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                (χ[1]-α[1])*(α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(δ[2]-α[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᶠᵇᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (α[1]-χ[1])*(δ[2]-α[2])*(β[3]-α[3]) +
                                                                (δ[1]-α[1])*(β[2]-α[2])*(α[3]-χ[3]) - (δ[1]-α[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                (α[1]-χ[1])*(β[2]-α[2])*(1+δ[3]-α[3]) - (1+β[1]-α[1])*(δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᵇᵇᶠ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+δ[3]-α[3]) + (α[1]-χ[1])*(δ[2]-α[2])*(α[3]-β[3]) +
                                                                (δ[1]-α[1])*(α[2]-β[2])*(α[3]-χ[3]) - (δ[1]-α[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                (α[1]-χ[1])*(α[2]-β[2])*(1+δ[3]-α[3]) - (1+α[1]-β[1])*(δ[2]-α[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᶠᶠᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (χ[1]-α[1])*(α[2]-δ[2])*(β[3]-α[3]) +
                                                                (α[1]-δ[1])*(β[2]-α[2])*(χ[3]-α[3]) - (α[1]-δ[1])*(1+χ[2]-α[2])*(β[3]-α[3]) -
                                                                (χ[1]-α[1])*(β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᵇᶠᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+χ[2]-α[2])*(1+α[3]-δ[3]) + (χ[1]-α[1])*(α[2]-δ[2])*(α[3]-β[3]) +
                                                                (α[1]-δ[1])*(α[2]-β[2])*(χ[3]-α[3]) - (α[1]-δ[1])*(1+χ[2]-α[2])*(α[3]-β[3]) -
                                                                (χ[1]-α[1])*(α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(α[2]-δ[2])*(χ[3]-α[3])) > 0 ? 1.0 : 0.0

jᶠᵇᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+β[1]-α[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (α[1]-χ[1])*(α[2]-δ[2])*(β[3]-α[3]) +
                                                                (α[1]-δ[1])*(β[2]-α[2])*(α[3]-χ[3]) - (α[1]-δ[1])*(1+α[2]-χ[2])*(β[3]-α[3]) -
                                                                (α[1]-χ[1])*(β[2]-α[2])*(1+α[3]-δ[3]) - (1+β[1]-α[1])*(α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0

jᵇᵇᵇ{N}(α::NTuple{N},β::NTuple{N},χ::NTuple{N},δ::NTuple{N}) = ((1+α[1]-β[1])*(1+α[2]-χ[2])*(1+α[3]-δ[3]) + (α[1]-χ[1])*(α[2]-δ[2])*(α[3]-β[3]) +
                                                                (α[1]-δ[1])*(α[2]-β[2])*(α[3]-χ[3]) - (α[1]-δ[1])*(1+α[2]-χ[2])*(α[3]-β[3]) -
                                                                (α[1]-χ[1])*(α[2]-β[2])*(1+α[3]-δ[3]) - (1+α[1]-β[1])*(α[2]-δ[2])*(α[3]-χ[3])) > 0 ? 1.0 : 0.0
