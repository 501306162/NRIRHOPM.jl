"""
Root abstract type for multi-dispatching regularization(smooth) terms
"""
abstract AbstractRegularization

"""
Potts Model(not implemented yet)
"""
type Potts <: AbstractRegularization
end

"""
Truncated Absolute Difference
"""
type TAD <: AbstractRegularization
end

"""
Quadratic Model(not implemented yet)
"""
type Quadratic <: AbstractRegularization
end


"""
Method for Truncated Absolute Difference
"""
function pairwiseclique(
    imageDims::Tuple{Int,Int},              # image dimentions
    deformers::Vector{Vector{Int}},         # transform vectors
    regularization::TAD;                    # regularization method
    γ::Real=1,                              # the rate of increase in the cost
    τ::Real=Inf,                            # controls when the cost stops increasing
    )
    # get the total number of pixels in the image and the number of transform vectors in deformers
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen)

    # pre-allocation
    # 8-neighborhood system:
    # {(x-2)×(y-2)×8 + [2×(x-2)+2×(y-2)]×5 + 4×3} × deformLen²
    valsLen = (8*(imageDims[1]-2)*(imageDims[2]-2) + 5*(2*(imageDims[1]+imageDims[2])-8) + 4*3) * deformLen * deformLen
    @show valsLen
    vals = zeros(Float64, valsLen)
    pos = zeros(Int, (2,valsLen))
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
                    vals[indexNum] = -truncated_absolute_diff(deformers[a], deformers[b]; c=Float64(γ), d=Float64(τ))
                    indTemp = sub2ind(tensorDimsIntermediate, i, a, j, b)
                    subTemp = ind2sub(tensorDimsSymmetric, indTemp)
                    pos[1, indexNum] = subTemp[1]
                    pos[2, indexNum] = subTemp[2]
                end
            end
        end
    end

    # force tensor₁ non-negative
    vals -= 1.1minimum(vals)

    return SparseArray(vals, pos, tensorDimsSymmetric)
end

"""
Construct the second-order potential tensor(matrix), also called **smooth term** or **regular term**.

Requires arguments:

- fixedImg::Array{T,N}                               # fixed(target) image
- movingImg::Array{T,N}                              # moving(source) image
- deformableWindow::Matrix{Vector{Int}}              # transform matrix
- regularization::AbstractRegularization = TAD()     # keyword argument for regularization selection
- γ::Real                                            # the rate of increase in the cost, argument for TAD
- τ::Real                                            # controls when the cost stops increasing, argument for TAD

The default regularization is TAD(Truncated Absolute Difference).
"""
function pairwiseclique{T,N}(
    fixedImg::Array{T,N},                              # fixed(target) image
    movingImg::Array{T,N},                             # moving(source) image
    deformableWindow::Matrix{Vector{Int}};             # transform matrix
    regularization::AbstractRegularization = TAD(),    # regularization selection
    γ::Real=1,                                         # [TAD] the rate of increase in the cost
    τ::Real=Inf,                                       # [TAD] controls when the cost stops increasing
    )
    # get image dimensions and checkout whether fixedImg and movingImg are of the same size
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image Dimension Mismatch!"))

    # ~~make deformers of type Vector{Tuple{Int,Int}} for accelerating~~
    deformers = reshape(deformableWindow, length(deformableWindow))
    # deformers = [tuple(v...) for v in deformers]

    # call corresponding methods
    info("Calling pairwiseclique:")
    if regularization == TAD()
        info("Regularization: TAD(Truncated Absolute Difference)")
        γ == 1 && info("You may need to specify the parameter γ when using TAD as regularization. The default value is 1, which means no scaling.")
        τ == Inf && info("You may need to specify the parameter τ when using TAD as regularization. The default value is Inf, which means no truncation.")
        return pairwiseclique(imageDims, deformers, regularization; γ=γ, τ=τ)
    else
        error("The implementation of $(regularization) is missing.")
    end
end
