function pairwiseclique4validation(imageDims, deformers)
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    # set up tensor dimensions
    # 𝐇⁴: forth-order tensor 𝐇ᵢₐⱼᵦ
    # 𝐇²: second-order symmetric tensor 𝐇ᵢⱼ
    𝐇⁴Dims = (imageLen, deformLen, imageLen, deformLen)
    𝐇²Dims = (imageLen * deformLen, imageLen * deformLen)

    # 8-neighborhood system
    # since 𝐇² is symmetric, we only consider the following cliques:
    #   ▦ ▦ □      ▦                ▦        y,x -->
    #   ▦ ⬔ □  =>    ⬔   ▦ ⬔    ⬔   ⬔        |
    #   ▦ □ □                 ▦              ↓
    r, c = imageDims
    dataLen = ((r-2)*(c-2)*4 + (r-2+c-2)*5 + 6) * deformLen^2

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{2,Int}}(dataLen)
    idx = 0

    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    @inbounds for ii in pixelRange
        i = sub2ind(imageDims, ii[1], ii[2])
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj < ii
                j = sub2ind(imageDims, jj[1], jj[2])
                for a in eachindex(deformers), b in eachindex(deformers)
                    idx += 1
                    data[idx] = e^-NRIRHOPM.tad(deformers[a], deformers[b], 1.0, Inf)
                    index[idx] = ind2sub(𝐇²Dims, sub2ind(𝐇⁴Dims, i, a, j, b))
                end
            end
        end
    end
    return SSTensor(data, index, 𝐇²Dims)
end

function treyclique4validation(imageDims, deformers)
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    𝐇⁶Dims = (imageLen, deformLen, imageLen, deformLen, imageLen, deformLen)
    𝐇³Dims = (imageLen*deformLen, imageLen*deformLen, imageLen*deformLen)

    # 8-neighborhood system
    # since 𝐇³ is symmetric, it's equivalent to only take into account the
    # following 4-neighborhood system:           y
    #   □ ▦ □        ▦   ▦                       ↑
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ⬔ ▦   ▦ ⬔   ⬔ ▦         |
    #   □ ▦ □                    ▦   ▦     (x,y) +--> x
    r, c = imageDims
    dataLen = ((r-2)*(c-2)*4 + 2*(r-2+c-2)*2 + 4) * deformLen^3

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{3,Int}}(dataLen)
    indexNum = 0

    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
		i = sub2ind(imageDims, ii.I...)
		neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
		neighbor4v = CartesianIndex[]
		neighbor4h = CartesianIndex[]
		for nn in neighborRange
			if nn[1] == ii[1] && nn != ii
				push!(neighbor4v, nn)
			end
			if nn[2] == ii[2] && nn != ii
				push!(neighbor4h, nn)
			end
        end
        for jj in neighbor4h, kk in neighbor4v
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            for b in eachindex(deformers), a in eachindex(deformers), c in eachindex(deformers)
                indexNum += 1
                cost = NRIRHOPM.topology_preserving([jj.I...], [ii.I...], [kk.I...], deformers[b], deformers[a], deformers[c])
                data[indexNum] = cost
                indTemp = sub2ind(𝐇⁶Dims, i, a, j, b, k, c)
                index[indexNum] = ind2sub(𝐇³Dims, indTemp)
            end
        end
    end
    return SSTensor(data, index, 𝐇³Dims)
end
