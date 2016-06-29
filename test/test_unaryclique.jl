using TensorDecompositions
using Base.Test

# data term
function unaryclique{T,N}(fixedImg::Array{T,N},
                          movingImg::Array{T,N},
                          deformers::Vector{Tuple{Int,Int}};
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

sad{T<:Number}(x::T, y::T) = abs(x-y)

function foo()
    deformableWindow = [[(i,j) for i in -1:1, j in -1:1]...]
    img = rand(32, 32)
    moving = rand(32, 32)
    @time out = unaryclique(img, moving, deformableWindow)
end

# warm-up
foo()

foo()
