"""
Calculates the sum of absolute differences between fixed(target) image and moving(source) image.
Returns the data term tensor(vector).
"""
function sum_absolute_diff{T,N}(
    fixedImg::Array{T,N},                    # fixed(target) image
    movingImg::Array{T,N},                   # moving(source) image
    deformers::Vector{Vector{Int}},          # transform vectors
    )
    # get image dimensions and checkout whether fixedImg and movingImg are of the same size
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image Dimension Mismatch!"))

    # get the total number of pixels in the image and the number of transform vectors in deformers
    imageLen = length(fixedImg)
    deformLen = length(deformers)

    # pre-allocation tensor₁
    tensor₁ = zeros(T, imageLen, deformLen)

    # calculate each entry of the tensor₁
    pixelRange = CartesianRange(imageDims)
    for ii in pixelRange, a in eachindex(deformers)
        i = sub2ind(imageDims, ii.I...)
        # ϕ(i) = i + d(i)
        ϕᵢᵢ = collect(ii.I) + deformers[a]
        if CartesianIndex(tuple(ϕᵢᵢ...)) ∈ pixelRange
            ϕᵢ = sub2ind(imageDims, ϕᵢᵢ...)
            tensor₁[i,a] = -abs(fixedImg[i] - movingImg[ϕᵢ])
        else
            tensor₁[i,a] = Inf
        end
    end

    # force tensor₁ non-negative
    tensor₁ -= 1.1minimum(tensor₁)
    tensor₁[tensor₁.==Inf] = 0
    
    return reshape(tensor₁, imageLen * deformLen)
end
