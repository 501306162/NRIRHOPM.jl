abstract Neighborhood{ImageDim,CliqueSize}

# common neighborhood systems
abstract Connected4{CliqueSize} <: Neighborhood{2,CliqueSize}
abstract Connected8{CliqueSize} <: Neighborhood{2,CliqueSize}
abstract Connected6{CliqueSize} <: Neighborhood{3,CliqueSize}
abstract Connected26{CliqueSize} <: Neighborhood{3,CliqueSize}

typealias SquareCubic Union{Connected8{2}, Connected26{2}}

type C8Pairwise <: Connected8{2} end
type C26Pairwise <: Connected26{2} end
type C8Topology <: Connected8{3} end
type C26Topology <: Connected26{4} end

typealias CnTopology Union{C8Topology, C26Topology}

"""
    neighbors(C8Pairwise(), imageDims) -> Vector{idxs}
    neighbors(C26Pairwise(), imageDims) -> Vector{idxs}
"""
function neighbors{T<:SquareCubic}(::T, imageDims)
    idx = NTuple{2,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for 𝒊 in pixelRange
        i = sub2ind(imageDims, 𝒊.I...)
        neighborRange = CartesianRange(max(pixelFirst, 𝒊-pixelFirst), min(pixelEnd, 𝒊+pixelFirst))
        for 𝒋 in neighborRange
            if 𝒋 < 𝒊
                j = sub2ind(imageDims, 𝒋.I...)
                push!(idx, (i,j))
            end
        end
    end
    return idx
end

"""
    neighbors(C8Topology(), imageDims) -> Vector{idxs}
"""
function neighbors(::C8Topology, imageDims::NTuple{2,Int})
    # 8-Connected neighborhood for 3-element cliques
    # since the tensor is symmetric, we only consider the following cliques:
    #   □ ⬓ □        ⬓                ⬓      r,c-->    ⬔ => 𝒊 => p1 => α
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ⬓ => 𝒋 => p2 => β
    #   □ ⬓ □              ⬓    ⬓            ↓         ▦ => 𝒌 => p3 => χ
    #              Jᵇᵇ   Jᶠᵇ    Jᶠᶠ   Jᵇᶠ
    idxJᶠᶠ = NTuple{3,Int}[]
    idxJᵇᶠ = NTuple{3,Int}[]
    idxJᶠᵇ = NTuple{3,Int}[]
    idxJᵇᵇ = NTuple{3,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for 𝒊 in pixelRange
        i = sub2ind(imageDims, 𝒊.I...)
        neighborRange = CartesianRange(max(pixelFirst, 𝒊-pixelFirst), min(pixelEnd, 𝒊+pixelFirst))

        𝒋 = 𝒊 + CartesianIndex(1,0)
        𝒌 = 𝒊 + CartesianIndex(0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            push!(idxJᶠᶠ, (i,j,k))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0)
        𝒌 = 𝒊 + CartesianIndex(0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            push!(idxJᵇᶠ, (i,j,k))
        end

        𝒋 = 𝒊 + CartesianIndex(1,0)
        𝒌 = 𝒊 - CartesianIndex(0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            push!(idxJᶠᵇ, (i,j,k))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0)
        𝒌 = 𝒊 - CartesianIndex(0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            push!(idxJᵇᵇ, (i,j,k))
        end
    end
    return [idxJᶠᶠ, idxJᵇᶠ, idxJᶠᵇ, idxJᵇᵇ]
end

"""
    neighbors(C26Topology(), imageDims) -> Vector{idxs}
"""
function neighbors(::C26Topology, imageDims::NTuple{3,Int})
    # 26-Connected neighborhood for 4-element cliques
    # coordinate system(r,c,z):
    #  up  r     c --->        z × × (front to back)
    #  to  |   left to right     × ×
    # down ↓
    # coordinate => point => label:
    # 𝒊 => p1 => α   𝒋 => p2 => β   𝒌 => p3 => χ   𝒎 => p5 => δ
    idxJᶠᶠᶠ = NTuple{4,Int}[]
    idxJᵇᶠᶠ = NTuple{4,Int}[]
    idxJᶠᵇᶠ = NTuple{4,Int}[]
    idxJᵇᵇᶠ = NTuple{4,Int}[]
    idxJᶠᶠᵇ = NTuple{4,Int}[]
    idxJᵇᶠᵇ = NTuple{4,Int}[]
    idxJᶠᵇᵇ = NTuple{4,Int}[]
    idxJᵇᵇᵇ = NTuple{4,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for 𝒊 in pixelRange
        i = sub2ind(imageDims, 𝒊.I...)
        neighborRange = CartesianRange(max(pixelFirst, 𝒊-pixelFirst), min(pixelEnd, 𝒊+pixelFirst))

        𝒋 = 𝒊 + CartesianIndex(1,0,0)
        𝒌 = 𝒊 + CartesianIndex(0,1,0)
        𝒎 = 𝒊 + CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᶠᶠᶠ, (i,j,k,m))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0,0)
        𝒌 = 𝒊 + CartesianIndex(0,1,0)
        𝒎 = 𝒊 + CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᵇᶠᶠ, (i,j,k,m))
        end

        𝒋 = 𝒊 + CartesianIndex(1,0,0)
        𝒌 = 𝒊 - CartesianIndex(0,1,0)
        𝒎 = 𝒊 + CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᶠᵇᶠ, (i,j,k,m))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0,0)
        𝒌 = 𝒊 - CartesianIndex(0,1,0)
        𝒎 = 𝒊 + CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᵇᵇᶠ, (i,j,k,m))
        end

        𝒋 = 𝒊 + CartesianIndex(1,0,0)
        𝒌 = 𝒊 + CartesianIndex(0,1,0)
        𝒎 = 𝒊 - CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᶠᶠᵇ, (i,j,k,m))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0,0)
        𝒌 = 𝒊 + CartesianIndex(0,1,0)
        𝒎 = 𝒊 - CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᵇᶠᵇ, (i,j,k,m))
        end

        𝒋 = 𝒊 + CartesianIndex(1,0,0)
        𝒌 = 𝒊 - CartesianIndex(0,1,0)
        𝒎 = 𝒊 - CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᶠᵇᵇ, (i,j,k,m))
        end

        𝒋 = 𝒊 - CartesianIndex(1,0,0)
        𝒌 = 𝒊 - CartesianIndex(0,1,0)
        𝒎 = 𝒊 - CartesianIndex(0,0,1)
        if 𝒋 in neighborRange && 𝒌 in neighborRange && 𝒎 in neighborRange
            j = sub2ind(imageDims, 𝒋.I...)
            k = sub2ind(imageDims, 𝒌.I...)
            m = sub2ind(imageDims, 𝒎.I...)
            push!(idxJᵇᵇᵇ, (i,j,k,m))
        end
    end
    return [idxJᶠᶠᶠ, idxJᵇᶠᶠ, idxJᶠᵇᶠ, idxJᵇᵇᶠ, idxJᶠᶠᵇ, idxJᵇᶠᵇ, idxJᶠᵇᵇ, idxJᵇᵇᵇ]
end
