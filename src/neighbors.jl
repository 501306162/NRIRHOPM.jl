# types for multi-dispatching
abstract Neighborhood{Dimension,CliqueSize}

abstract Connected4{CliqueSize} <: Neighborhood{2,CliqueSize}

abstract Connected8{CliqueSize} <: Neighborhood{2,CliqueSize}

abstract Connected6{CliqueSize} <: Neighborhood{3,CliqueSize}

abstract Connected26{CliqueSize} <: Neighborhood{3,CliqueSize}

typealias SquareCubic Union{Connected8{2}, Connected26{2}}

function neighbors{T<:SquareCubic}(::Type{T}, imageDims)
    idx = NTuple{2,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for 𝒊 in pixelRange
        i = sub2ind(imageDims, 𝒊.I...)
        neighborRange = CartesianRange(max(pixelFirst, 𝒊-pixelFirst), min(pixelEnd, 𝒊+pixelFirst))
        for 𝐣 in neighborRange
            if 𝐣 < 𝒊
                j = sub2ind(imageDims, 𝐣.I...)
                push!(idx, (i,j))
            end
        end
    end
    return idx
end

function neighbors(::Type{Connected8{3}}, imageDims::NTuple{2,Int})
    # 8-Connected neighborhood for 3-element cliques
    # since the tensor is symmetric, we only consider the following cliques:
    #   □ ⬓ □        ⬓                ⬓      r,c-->    ⬔ => ii => p1 => α
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ⬓ => jj => p2 => β
    #   □ ⬓ □              ⬓    ⬓            ↓         ▦ => kk => p3 => χ
    #              Jᵇᵇ   Jᶠᵇ    Jᶠᶠ   Jᵇᶠ
    idxJᶠᶠ = NTuple{3,Int}[]
    idxJᵇᶠ = NTuple{3,Int}[]
    idxJᶠᵇ = NTuple{3,Int}[]
    idxJᵇᵇ = NTuple{3,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii.I...)
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))

        jj = ii + CartesianIndex(1,0)
        kk = ii + CartesianIndex(0,1)
        if jj in neighborRange && kk in neighborRange
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            push!(idxJᶠᶠ, (i,j,k))
        end

        jj = ii - CartesianIndex(1,0)
        kk = ii + CartesianIndex(0,1)
        if jj in neighborRange && kk in neighborRange
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            push!(idxJᵇᶠ, (i,j,k))
        end

        jj = ii + CartesianIndex(1,0)
        kk = ii - CartesianIndex(0,1)
        if jj in neighborRange && kk in neighborRange
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            push!(idxJᶠᵇ, (i,j,k))
        end

        jj = ii - CartesianIndex(1,0)
        kk = ii - CartesianIndex(0,1)
        if jj in neighborRange && kk in neighborRange
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            push!(idxJᵇᵇ, (i,j,k))
        end
    end
    return idxJᶠᶠ, idxJᵇᶠ, idxJᶠᵇ, idxJᵇᵇ
end

function neighbors(::Type{Connected26{4}}, imageDims::NTuple{3,Int})
    # 26-Connected neighborhood for 4-element cliques
    # coordinate system(r,c,z):
    #  up  r     c --->        z × × (front to back)
    #  to  |   left to right     × ×
    # down ↓
    # coordinate => point => label:
    # iii => p1 => α   jjj => p2 => β   kkk => p3 => χ   mmm => p5 => δ
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
    for iii in pixelRange
        i = sub2ind(imageDims, iii.I...)
        neighborRange = CartesianRange(max(pixelFirst, iii-pixelFirst), min(pixelEnd, iii+pixelFirst))

        jjj = iii + CartesianIndex(1,0,0)
        kkk = iii + CartesianIndex(0,1,0)
        mmm = iii + CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᶠᶠᶠ, (i,j,k,m))
        end

        jjj = iii - CartesianIndex(1,0,0)
        kkk = iii + CartesianIndex(0,1,0)
        mmm = iii + CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᵇᶠᶠ, (i,j,k,m))
        end

        jjj = iii + CartesianIndex(1,0,0)
        kkk = iii - CartesianIndex(0,1,0)
        mmm = iii + CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᶠᵇᶠ, (i,j,k,m))
        end

        jjj = iii - CartesianIndex(1,0,0)
        kkk = iii - CartesianIndex(0,1,0)
        mmm = iii + CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᵇᵇᶠ, (i,j,k,m))
        end

        jjj = iii + CartesianIndex(1,0,0)
        kkk = iii + CartesianIndex(0,1,0)
        mmm = iii - CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᶠᶠᵇ, (i,j,k,m))
        end

        jjj = iii - CartesianIndex(1,0,0)
        kkk = iii + CartesianIndex(0,1,0)
        mmm = iii - CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᵇᶠᵇ, (i,j,k,m))
        end

        jjj = iii + CartesianIndex(1,0,0)
        kkk = iii - CartesianIndex(0,1,0)
        mmm = iii - CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᶠᵇᵇ, (i,j,k,m))
        end

        jjj = iii - CartesianIndex(1,0,0)
        kkk = iii - CartesianIndex(0,1,0)
        mmm = iii - CartesianIndex(0,0,1)
        if jjj in neighborRange && kkk in neighborRange && mmm in neighborRange
            j = sub2ind(imageDims, jjj.I...)
            k = sub2ind(imageDims, kkk.I...)
            m = sub2ind(imageDims, mmm.I...)
            push!(idxJᵇᵇᵇ, (i,j,k,m))
        end
    end
    return idxJᶠᶠᶠ, idxJᵇᶠᶠ, idxJᶠᵇᶠ, idxJᵇᵇᶠ, idxJᶠᶠᵇ, idxJᵇᶠᵇ, idxJᶠᵇᵇ, idxJᵇᵇᵇ
end
