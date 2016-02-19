# functions for generating "cliquelized" potentials

# setup unary-clique potential (data term)
function unaryclique{T,N}(fixedImg::Array{T,N},
                          movingImg::Array{T,N},
                          deformers::Vector{Tuple{Int64,Int64}};
                          δ::Float64=1e2
                         )
    # get image dimensions and checkout whether fixedImg and movingImg are of the same size
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image dimensions mismatch."))
    # get the total number of pixels in the image and the number of transform vectors in deformers
    imageLen = length(fixedImg)
    deformLen = length(deformers)
    # pre-allocation tensorH
    tensorH = zeros(T, imageLen, deformLen)
    # calculate each entry of the tensorH
    for i in eachindex(fixedImg), a in eachindex(deformers)
        # i + d(i) => (x,y) + deformerᵢ => (x,y) + deformers[a]
        xy = ind2sub(imageDims, i)
        xyDeformed = (xy[1] + deformers[a][1], xy[2] + deformers[a][2])
        iDeformed = sub2ind(imageDims, xyDeformed...)
        # calculate potentials and punish those transform faults with "-∞"
        if iDeformed > 0 && iDeformed <= imageLen
            tensorH[i,a] = -sad(fixedImg[i], movingImg[iDeformed])
        else
            tensorH[i,a] = Inf
        end
    end
    minH = minimum(tensorH)
    tensorH -= (minH - δ)
    tensorH[tensorH.==Inf] = 0
    return reshape(tensorH, imageLen * deformLen)
end


# setup pair-wise clique potential (smooth term)
function pairwiseclique(imageDims::Tuple{Int64, Int64}, deformers::Vector{Tuple{Int64,Int64}}; δ::Float64=1e2)
    # get the total number of pixels in the image and the number of transform vectors in deformers
    deformLen = length(deformers)
    imageLen = prod(imageDims)
    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen)
    # pre-calculate tensorH and cost
    tensorH = zeros(tensorDimsIntermediate)
    fill!(tensorH, Inf)
    cost = zeros(deformLen, deformLen)
    for a in eachindex(deformers), b in eachindex(deformers)
        cost[a,b] = -distance(deformers[a], deformers[b])
    end
    costTensor = reshape(cost, 1, deformLen, 1, deformLen)
    # neighborhood filter
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii.I...)
        neighbors = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighbors
            if jj != ii
                j = sub2ind(imageDims, jj.I...)
                tensorH[i,:,j,:] = costTensor
            end
        end
    end
    minH = minimum(tensorH)
    tensorH -= (minH - δ)
    tensorH[tensorH.==Inf] = 0
    return reshape(tensorH, tensorDimsSymmetric)
end

function pairwiseclique2{T,N}(fixedImg::Array{T,N},
                              movingImg::Array{T,N},
                              deformers::Matrix{Vector{Int64}};
                              δ::Float64=1e2
                             )
    # get image dimensions and checkout whether fixedImg and movingImg are of the same size
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image dimensions mismatch."))
    # get the total number of pixels in the image and the number of transform vectors in deformers
    deformLen = length(deformers)
    imageLen = length(fixedImg)
    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen)
    # pre-allocation
    # 8-neighborhood system:
    # {(x-2)×(y-2)×8 + [2×(x-2)+2×(y-2)]×5 + 4×3} × deformLen²
    valsLen = (8*(imageDims[1]-2)*(imageDims[2]-2) + 5*(2*(imageDims[1]+imageDims[2])-8) + 4*3) * deformLen * deformLen
    @show valsLen
    vals = zeros(Float64, valsLen)
    pos = zeros(Int64, 2*valsLen)
    indexNum = 0
    # neighborhood filter
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii.I...)
        neighbors = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighbors
            if jj != ii
                j = sub2ind(imageDims, jj.I...)
                for a in eachindex(deformers), b in eachindex(deformers)
                    indexNum += 1
                    vals[indexNum] = -distance(deformers[a], deformers[b])
                    indTemp = sub2ind(tensorDimsIntermediate, i, a, j, b)
                    subTemp = ind2sub(tensorDimsSymmetric, indTemp)
                    pos[2indexNum-1] = subTemp[1]
                    pos[2indexNum] = subTemp[2]
                end
            end
        end
    end
    minVals = minimum(vals)
    vals -= (minVals - δ)
    SparseArray(vals, reshape(pos, 2, length(vals)), tensorDimsSymmetric)
end

# setup 3rd-element clique potential (topology preserving term)
function treyclique(imageDims::Tuple{Int64, Int64}, deformers::Vector{Tuple{Int64, Int64}})
    deformLen = length(deformers)
    imageLen = prod(imageDims)
    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen, imageLen*deformLen)
    # neighborhood filter
    deep = 0
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    pos = Int64[]
    vals = Float64[]
    for ii in pixelRange
		i = sub2ind(imageDims, ii.I...)
		neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
		neighbor4h = CartesianIndex[]
		neighbor4v = CartesianIndex[]
		for nn in neighborRange
			if nn[1] == ii[1] && nn != ii
				push!(neighbor4h, nn)
			end
			if nn[2] == ii[2] && nn != ii
				push!(neighbor4v, nn)
			end
			deep += 1
        end
        for jj in neighbor4h, kk in neighbor4v
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            for a in eachindex(deformers), b in eachindex(deformers), c in eachindex(deformers)
                cost = topology(ii.I, jj.I, kk.I, deformers[a], deformers[b], deformers[c])
                if cost <= 0
                    push!(vals, 1,1,1,1,1,1)
                    indTemp = sub2ind(tensorDimsIntermediate, i, a, j, b, k, c)
                    subTemp = ind2sub(tensorDimsSymmetric, indTemp)
                    append!(pos, [collect(permutations(subTemp))...;])
                    deep += 1
                end
            end
        end
    end
    @show deep
    SparseArray(vals, reshape(pos, 3, length(vals)), tensorDimsSymmetric)
end
