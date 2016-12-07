"""
    pairwiseclique(fixedImg, movingImg, deformableWindow; <keyword arguments>)

Construct the second-order tensor, also called **smooth term** or **regular term**.

# Arguments
* `fixedImg::Array{T,N}`: the fixed(target) image.
* `movingImg::Array{T,N}`: the moving(source) image.
* `deformableWindow::Matrix{Vector{Int}}`: the transform matrix.
* `algorithm::SmoothTerm=TAD()`: the method for calculating smooth cost.
* `ω::Real=1`: the weighted parameter.
* `χ::Real=1`: the rate of increase in the cost, argument for TAD.
* `δ::Real=Inf`: controls when the cost stops increasing, argument for TAD.
"""
function pairwiseclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}};
    algorithm::SmoothTerm=TAD(),
    ω::Real=1,
    χ::Real=1,
    δ::Real=Inf
    )
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Image Dimension Mismatch!"))

    deformers = reshape(deformableWindow, length(deformableWindow))
    deformers = [tuple(v...) for v in deformers]

    info("Calling pairwiseclique:")
    if algorithm == TAD()
        info("Algorithm: TAD(Truncated Absolute Difference)")
        return pairwiseclique(imageDims, deformers, algorithm, Float64(ω), Float64(χ), Float64(δ))
    else
        throw(ArgumentError("The implementation of $(algorithm) is missing."))
    end
end

"""
    pairwiseclique(imageDims, deformers, TAD()[, 1.0, 1.0, Inf]) -> PSSTensor

The method for the Truncated Absolute Difference(TAD). Returns a `PSSTensor` 𝐇².

# Arguments
* `imageDims::NTuple{2,Ti}`: the size of the 2D image.
* `deformers::Vector{NTuple{N,Td}}`: transform vectors.
* `algorithm::TAD`: the method for calculating smooth cost.
* `ω::Float64`: the weighted parameter.
* `χ::Float64`: the rate of increase in the cost.
* `δ::Float64`: controls when the cost stops increasing.
"""
function pairwiseclique{Ti<:Integer,Td,N}(
    imageDims::NTuple{N,Ti},
    deformers::Vector{NTuple{N,Td}},
    algorithm::TAD,
    ω::Float64=1.0,
    χ::Float64=1.0,
    δ::Float64=Inf
    )
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
    @inbounds for ii in pixelRange
        i = sub2ind(imageDims, ii[1], ii[2])
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj < ii
                j = sub2ind(imageDims, jj[1], jj[2])
                for a in eachindex(deformers), b in eachindex(deformers)
                    indexNum += 1
                    data[indexNum] = e^-truncated_absolute_diff(deformers[a], deformers[b], χ, δ)
                    index[indexNum] = ind2sub(𝐇²Dims, sub2ind(𝐇⁴Dims, i, a, j, b))
                end
            end
        end
    end
    return PSSTensor(ω*data, index, 𝐇²Dims)
end
