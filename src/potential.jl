# types for multi-dispatching
abstract AbstractPotential{Order}

# various dialects
typealias UnaryPotential AbstractPotential{1}
typealias DataTerm AbstractPotential{1}
typealias DataCost AbstractPotential{1}

typealias PairwisePotential AbstractPotential{2}
typealias SmoothTerm AbstractPotential{2}
typealias RegularTerm AbstractPotential{2}

typealias TreyPotential AbstractPotential{3}

# unary potentials
"""
Sum of Absolute Differences
"""
type SAD <: DataCost
end

"""
    sum_absolute_diff(fixedImg, movingImg, deformers) -> Vector

Calculates the sum of absolute differences between fixed(target) image
and moving(source) image. Returns a data term tensor(vector).

# Arguments
* `fixedImg::Array{T,N}`: the fixed(target) image.
* `movingImg::Array{T,N}`: the moving(source) image.
* `deformers::Vector{Vector{Int}}`: the transform vectors.
"""
function sum_absolute_diff{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformers::Vector{Vector{Int}}
    )
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image Dimension Mismatch!"))

    imageLen = length(fixedImg)
    deformLen = length(deformers)

    𝐇¹ = zeros(T, imageLen, deformLen)

    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange, a in eachindex(deformers)
        i = sub2ind(imageDims, ii[1], ii[2])
        # ϕ(x,y) = i(x,y) + d(x,y)
        ϕxᵢᵢ = ii[1] + deformers[a][1]
        ϕyᵢᵢ = ii[2] + deformers[a][2]
        if pixelFirst[1] <= ϕxᵢᵢ <= pixelEnd[1] && pixelFirst[2] <= ϕyᵢᵢ <= pixelEnd[2]
            ϕᵢ = sub2ind(imageDims, ϕxᵢᵢ, ϕyᵢᵢ)
            𝐇¹[i,a] = e^-abs(fixedImg[i] - movingImg[ϕᵢ])
        else
            𝐇¹[i,a] = 0
        end
    end
    return reshape(𝐇¹, imageLen * deformLen)
end

"""
Mutual Information
"""
type MI <: DataCost
end


# pairwise potentials
"""
Potts Model
"""
type Potts <: SmoothTerm
end

"""
Truncated Absolute Difference
"""
type TAD <: SmoothTerm
end

"""
    truncated_absolute_diff(fp, fq, c, d) -> Float64

Calculates the truncated absolute difference between two transform vectors.
Returns the cost value.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation
for early vision." International journal of computer vision 70.1 (2006): 41-54.

# Arguments
* `fp::NTuple{N,Ti}`: the transform vector(label) at pixel p.
* `fq::NTuple{N,Ti}`: the transform vector(label) at pixel q.
* `c::Float64`: the rate of increase in the cost.
* `d::Float64`: controls when the cost stops increasing.
"""
@generated function truncated_absolute_diff{Ti,N}(fp::NTuple{N,Ti}, fq::NTuple{N,Ti}, c::Float64, d::Float64)
    ex = :(0)
    for i = 1:N
        ex = :(abs2(fp[$i]-fq[$i]) + $ex)
    end
    return :(min(c * abs(√$ex), d))
end

"""
Quadratic Model
"""
type Quadratic <: SmoothTerm
end

# high-order potentials
"""
Topology Preservation
"""
type TP <: TreyPotential
end

"""
    topology_preserving(s₁, s₂, s₃, a, b, c) -> Int

Returns the cost value.

Refer to the following paper for further details:

Cordero-Grande, Lucilio, et al. "A Markov random field approach for
topology-preserving registration: Application to object-based tomographic image
interpolation." IEEE Transactions on Image Processing 21.4 (2012): 2047-2061.
"""
@inline function topology_preserving{T<:Integer}(s₁::Vector{T}, s₂::Vector{T}, s₃::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T})
    @inbounds begin
        𝐤s₁, 𝐤s₂, 𝐤s₃ = s₁ + a, s₂ + b, s₃ + c
        ∂φ₁∂φ₂ = (𝐤s₂[2] - 𝐤s₁[2]) * (𝐤s₂[1] - 𝐤s₃[1])
        ∂φ₂∂φ₁ = (𝐤s₂[1] - 𝐤s₁[1]) * (𝐤s₂[2] - 𝐤s₃[2])
        ∂r₁∂r₂ = (s₂[2] - s₁[2])*(s₂[1] - s₃[1])
    end
    v = (∂φ₁∂φ₂ - ∂φ₂∂φ₁) / ∂r₁∂r₂
    return v > 0 ? 0 : 1
end
