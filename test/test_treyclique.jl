using TensorDecompositions
using Base.Test

# topology preserving term
function treyclique(imageDims::Tuple{Int, Int}, deformers::Vector{Tuple{Int, Int}})
    deformLen = length(deformers)
    imageLen = prod(imageDims)
    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen, imageLen*deformLen)
    # neighborhood filter
    deep = 0
    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    pos = Int[]
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

function topology(s1::Tuple{Int,Int}, s2::Tuple{Int,Int}, s3::Tuple{Int,Int}, a::Tuple{Int,Int}, b::Tuple{Int,Int}, c::Tuple{Int,Int})
    ks1 = (s1[1]+a[1], s1[2]+a[2])
    ks2 = (s2[1]+b[1], s2[2]+b[2])
    ks3 = (s3[1]+c[1], s3[2]+c[2])
    dφ1 = ks2[1] - ks1[1]
    dr1 = s2[1] - s1[1]
    dφ2 = ks2[2] - ks3[2]
    dr2 = s2[2] - s3[2]
    dφ3 = ks2[2] - ks1[2]
    # dr3 = dr1
    dφ4 = ks2[1] - ks3[1]
    # dr4 = dr2
    v = (dφ1/dr1 * dφ2/dr2) - (dφ3/dr1 * dφ4/dr2)
    return v::Float64
end

function foo()
    deformableWindow = [[(i,j) for i in -2:2, j in -2:2]...]
    img = rand(32, 32)
    @time out = treyclique(size(img), deformableWindow)
end

# warm-up
foo()

foo()

