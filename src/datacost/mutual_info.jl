"""
Calculates the mutual information between fixed(target) image and moving(source) image.
Returns the data term tensor(vector).

Refer to the following paper for further details:

Maes, Frederik, et al. "Multimodality image registration by maximization of mutual information." IEEE transactions on Medical Imaging 16.2 (1997): 187-198.
"""
function mutual_info{T,N}(
    fixedImg::Array{T,N},                    # fixed(target) image
    movingImg::Array{T,N},                   # moving(source) image
    deformers::Vector{Vector{Int}},          # transform vectors
    β::Int                                   # number of bins used for histogram
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
            # do pixel-wise registration
            deformedImg = copy(movingImg)
            deformedImg[i] = movingImg[ϕᵢ]
            # compute the MI of this single step registration
            histFD = fit(Histogram, (reshape(fixedImg, imageLen), reshape(deformedImg, imageLen)), nbins=β)
            jointDistributionFD = histFD.weights/imageLen
            ηD = entropy(sum(jointDistributionFD, 1))
            ηF = entropy(sum(jointDistributionFD, 2))
            ηFD = entropy(jointDistributionFD)
            #=
            since only the eigenvector(indicator) is required, we can directly use
            the "single step MI" as an entry. consequently, the final result is not MI,
            but a "cumulative MI".
            =#
            tensor₁[i,a] = ηF + ηD - ηFD
        end
    end

    return reshape(tensor₁, imageLen * deformLen)
end
