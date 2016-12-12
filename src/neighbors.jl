# types for multi-dispatching
abstract Neighborhood{Dimension,CliqueSize}

type Connected4{CliqueSize} <: Neighborhood{2,CliqueSize}
end

type Connected8{CliqueSize} <: Neighborhood{2,CliqueSize}
end

type Connected6{CliqueSize} <: Neighborhood{3,CliqueSize}
end

type Connected26{CliqueSize} <: Neighborhood{3,CliqueSize}
end

function neighbors(::Type{Connected8{2}}, imageDims::NTuple{2,Int})
    # 8-Connected neighborhood for 2-element cliques
    # since the tensor is symmetric, we only consider the following cliques:
    #   ▦ ▦ □      ▦                ▦
    #   ▦ ⬔ □  =>    ⬔   ▦ ⬔    ⬔   ⬔
    #   ▦ □ □                 ▦
    idx = NTuple{2,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii[1], ii[2])
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj < ii
                j = sub2ind(imageDims, jj[1], jj[2])
                push!(idx, (i,j))
            end
        end
    end
    return idx
end

function neighbors(::Type{Connected8{3}}, imageDims::NTuple{2,Int})
    # 8-Connected neighborhood for 3-element cliques
    # since the tensor is symmetric, we only consider the following cliques:
    #   □ ⬓ □        ⬓                ⬓
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦
    #   □ ⬓ □              ⬓    ⬓
    idx = NTuple{3,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii.I...)
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj != ii && jj[2] == ii[2]
                j = sub2ind(imageDims, jj.I...)
                for kk in neighborRange
                    if kk != ii && kk[1] == ii[1]
                        k = sub2ind(imageDims, kk.I...)
                        push!(idx, (i,j,k))
                    end
                end
            end
        end
    end
    return idx
end

function neighbors(::Type{Connected26{2}}, imageDims::NTuple{3,Int})
    idx = NTuple{3,Int}[]
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for iii in pixelRange
        i = sub2ind(imageDims, iii[1], iii[2], iii[3])
        neighborRange = CartesianRange(max(pixelFirst, iii-pixelFirst), min(pixelEnd, iii+pixelFirst))
        for jjj in neighborRange
            if jjj < iii
                j = sub2ind(imageDims, jjj[1], jjj[2], jjj[3])
                push!(idx, (i,j))
            end
        end
    end
    return idx
end
