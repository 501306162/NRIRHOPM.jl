"""
Calculates the mutual information between fixed(target) image and moving(source) image.
Returns the data term tensor(vector).
"""
function mutual_info{T,N}(
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
    # deformedImg = copy(movingImg)
    # deformedImg[i] = movingImg[iDeformed]
    # hFD = fit(Histogram, (reshape(fixedImg, imageLen),reshape(deformedImg, imageLen)), nbins=binNum)
    # eFD = entropy(hFD.weights/binNum)
end
