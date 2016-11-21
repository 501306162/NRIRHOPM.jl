function treyclique(imageDims::Tuple{Int64, Int64}, deformers::Vector{NTuple{2,Int}})
    # set up tensor dimensions
    deformLen = length(deformers)
    imageLen = prod(imageDims)
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen, imageLen*deformLen)

    # init
    data = Float64[]
    index = NTuple{3,Int}[]

    # 8-neighborhood system
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
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
        end
        for jj in neighbor4h, kk in neighbor4v
            j = sub2ind(imageDims, jj.I...)
            k = sub2ind(imageDims, kk.I...)
            for a in eachindex(deformers), b in eachindex(deformers), c in eachindex(deformers)
                # calculate cost
                cost = norm([deformers[a]...;]) + norm([deformers[b]...;]) + norm([deformers[c]...;])
                push!(data, cost)
                # calculate index
                indTemp = sub2ind(tensorDimsIntermediate, i, a, j, b, k, c)
                subTemp = ind2sub(tensorDimsSymmetric, indTemp)
                push!(index, subTemp)
            end
        end
    end
    SparseSymmetric3WayTensor(data, index, tensorDimsSymmetric[1])
end
