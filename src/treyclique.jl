"""
    treyclique(fixedImg, movingImg, deformableWindow; <keyword arguments>)

Construct the third-order tensor.

# Arguments
* `fixedImg::Array{T,N}`: the fixed(target) image.
* `movingImg::Array{T,N}`: the moving(source) image.
* `deformableWindow::Matrix{Vector{Int}}`: the transform matrix.
* `algorithm::TreyPotential=TP()`: the method for calculating third order potential.
* `ω::Real=1`: the weighted parameter.
"""
function treyclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}};
    algorithm::TreyPotential=TP(),
    ω::Real=1
    )
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image Dimension Mismatch!"))

    deformers = reshape(deformableWindow, length(deformableWindow))

    info("Calling treyclique:")
    if algorithm == TP()
        info("Algorithm: TP(Topology Preserving)")
        return treyclique(imageDims, deformers, algorithm, Float64(ω))
    else
        throw(ArgumentError("The implementation of $(algorithm) is missing."))
    end
end

"""
    treyclique(imageDims, deformers, TP()[, 1.0]) -> PSSTensor

The method for the Topology Preserving(TP). Returns a `PSSTensor` 𝐇³.

# Arguments
* `imageDims::NTuple{2,Ti<:Integer}`: the size of the 2D image.
* `deformers::Vector{Vector{Ti<:Integer}}`: transform vectors.
* `algorithm::TP`: the method for calculating third order potential.
* `ω::Float64`: the weighted parameter.
"""
function treyclique{Ti<:Integer}(
    imageDims::NTuple{2,Ti},
    deformers::Vector{Vector{Ti}},
    algorithm::TP,
    ω::Float64=1.0
    )
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    𝐇⁶Dims = (imageLen, deformLen, imageLen, deformLen, imageLen, deformLen)
    𝐇³Dims = (imageLen*deformLen, imageLen*deformLen, imageLen*deformLen)

    # 8-neighborhood system
    # since 𝐇³ is symmetric, it's equivalent to only take into account the
    # following 4-neighborhood system:
    #   □ ▦ □        ▦   ▦
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ⬔ ▦   ▦ ⬔   ⬔ ▦
    #   □ ▦ □                    ▦   ▦
    x, y = imageDims
    dataLen = ((x-2)*(y-2)*4 + 2*(x-2+y-2)*2 + 4) * deformLen^3

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{3,Ti}}(dataLen)
    indexNum = 0

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
                indexNum += 1
                cost = topology_preserving([jj.I...], [ii.I...], [kk.I...], deformers[a], deformers[b], deformers[c])
                data[indexNum] = -cost
                indTemp = sub2ind(𝐇⁶Dims, i, a, j, b, k, c)
                index[indexNum] = ind2sub(𝐇³Dims, indTemp)
            end
        end
    end
    return PSSTensor(ω*data, index, 𝐇³Dims)
end
