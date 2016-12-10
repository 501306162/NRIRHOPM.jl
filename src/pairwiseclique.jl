"""
    pairwiseclique(fixedImg, movingImg, deformableWindow; <keyword arguments>)

Construct the second-order tensor, also called **smooth cost** or **regular term**.

# Arguments
* `fixedImg::Array{T,N}`: the fixed(target) image.
* `movingImg::Array{T,N}`: the moving(source) image.
* `deformableWindow::Matrix{Vector{Int}}`: the transform matrix.
* `algorithm::SmoothTerm=TAD()`: the method for calculating smooth cost.
* `ω::Real=1`: the weighted parameter.
* `χ::Real=1`: the rate of increase in the cost(argument for TAD & TQD).
* `δ::Real=Inf`: controls when the cost stops increasing(argument for TAD & TQD).
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
    if algorithm == Potts()
        info("Algorithm: Potts(Potts Model)")
        return pairwiseclique(imageDims, deformers, algorithm, ω)
    elseif algorithm == TAD()
        info("Algorithm: TAD(Truncated Absolute Difference)")
        return pairwiseclique(imageDims, deformers, algorithm, ω, χ, δ)
    elseif algorithm == TQD()
        info("Algorithm: TQD(Truncated Quadratic Difference)")
        return pairwiseclique(imageDims, deformers, algorithm, ω, χ, δ)
    else
        throw(ArgumentError("The implementation of $(algorithm) is missing."))
    end
end

"""
    pairwiseclique(imageDims, deformers, Potts()) -> PSSTensor
    pairwiseclique(imageDims, deformers, Potts(), 1) -> PSSTensor

The method for the Potts(Potts model). Returns a `PSSTensor` 𝐇².

# Arguments
* `imageDims::NTuple{2,Ti}`: the size of the 2D image.
* `deformers::Vector{NTuple{2}}`: the transform vectors.
* `algorithm::Potts`: the specific method for calculating smooth cost.
* `weight`: the weighted parameter, also the constant in Potts model.
"""
function pairwiseclique{Ti<:Integer}(
    imageDims::NTuple{2,Ti},
    deformers::Vector{NTuple{2}},
    algorithm::Potts,
    weight=1.0
    )
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    # set up tensor dimensions
    # 𝐇⁴: forth-order tensor 𝐇ᵢₐⱼᵦ
    # 𝐇²: second-order symmetric tensor 𝐇ᵢⱼ
    𝐇⁴Dims = (imageLen, deformLen, imageLen, deformLen)
    𝐇²Dims = (imageLen * deformLen, imageLen * deformLen)

    # 8-neighborhood system
    # since 𝐇² is symmetric, we only consider the following cliques:
    #   ▦ ▦ □      ▦                ▦
    #   ▦ ⬔ □  =>    ⬔   ▦ ⬔    ⬔   ⬔
    #   ▦ □ □                 ▦
    x, y = imageDims
    dataLen = ((x-2)*(y-2)*4 + (x-2+y-2)*5 + 6) * deformLen^2

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{2,Ti}}(dataLen)
    idx = 0

    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    for ii in pixelRange
        i = sub2ind(imageDims, ii.I...)
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj < ii
                j = sub2ind(imageDims, jj.I...)
                for a in eachindex(deformers), b in eachindex(deformers)
                    idx += 1
                    data[idx] = e^-potts_model(deformers[a], deformers[b], weight)
                    index[idx] = ind2sub(𝐇²Dims, sub2ind(𝐇⁴Dims, i, a, j, b))
                end
            end
        end
    end
    return PSSTensor(data, index, 𝐇²Dims)
end

"""
    pairwiseclique(imageDims, deformers, TAD()) -> PSSTensor
    pairwiseclique(imageDims, deformers, TAD(), 1, 1, Inf) -> PSSTensor

The method for the Truncated Absolute Difference(TAD). Returns a `PSSTensor` 𝐇².

# Arguments
* `imageDims::NTuple{2,Ti}`: the size of the 2D image.
* `deformers::Vector{NTuple{2}}`: the transform vectors.
* `algorithm::TAD`: the specific method for calculating smooth cost.
* `weight`: the weighted parameter.
* `rate`: the rate of increase in the cost.
* `threshold`: controls when the cost stops increasing.
"""
function pairwiseclique{Ti<:Integer}(
    imageDims::NTuple{2,Ti},
    deformers::Vector{NTuple{2}},
    algorithm::TAD,
    weight=1.0,
    rate=1.0,
    threshold=Inf
    )
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    # set up tensor dimensions
    tensorDimsIntermediate = (imageLen, deformLen, imageLen, deformLen)
    tensorDimsSymmetric = (imageLen*deformLen, imageLen*deformLen)

    # 8-neighborhood system
    # since 𝐇² is symmetric, we only consider the following cliques:
    #   ▦ ▦ □      ▦                ▦
    #   ▦ ⬔ □  =>    ⬔   ▦ ⬔    ⬔   ⬔
    #   ▦ □ □                 ▦
    x, y = imageDims
    dataLen = ((x-2)*(y-2)*4 + (x-2+y-2)*5 + 6) * deformLen^2

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{2,Ti}}(dataLen)
    idx = 0

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
                    idx += 1
                    data[idx] = e^-truncated_absolute_diff(deformers[a], deformers[b], rate, threshold)
                    index[idx] = ind2sub(𝐇²Dims, sub2ind(𝐇⁴Dims, i, a, j, b))
                end
            end
        end
    end
    return PSSTensor(weight*data, index, 𝐇²Dims)
end

"""
    pairwiseclique(imageDims, deformers, TQD()) -> PSSTensor
    pairwiseclique(imageDims, deformers, TQD(), 1, 1, Inf) -> PSSTensor

The method for the Truncated Quadratic Difference(TQD). Returns a `PSSTensor` 𝐇².

# Arguments
* `imageDims::NTuple{2,Ti}`: the size of the 2D image.
* `deformers::Vector{NTuple{2}}`: the transform vectors.
* `algorithm::TQD`: the specific method for calculating smooth cost.
* `weight`: the weighted parameter.
* `rate`: the rate of increase in the cost.
* `threshold`: controls when the cost stops increasing.
"""
function pairwiseclique{Ti<:Integer}(
    imageDims::NTuple{2,Ti},
    deformers::Vector{NTuple{2}},
    algorithm::TQD,
    weight=1.0,
    rate=1.0,
    threshold=Inf
    )
    deformLen = length(deformers)
    imageLen = prod(imageDims)

    # set up tensor dimensions
    # 𝐇⁴: forth-order tensor 𝐇ᵢₐⱼᵦ
    # 𝐇²: second-order symmetric tensor 𝐇ᵢⱼ
    𝐇⁴Dims = (imageLen, deformLen, imageLen, deformLen)
    𝐇²Dims = (imageLen * deformLen, imageLen * deformLen)

    # 8-neighborhood system
    # since 𝐇² is symmetric, we only consider the following cliques:
    #   ▦ ▦ □      ▦                ▦
    #   ▦ ⬔ □  =>    ⬔   ▦ ⬔    ⬔   ⬔
    #   ▦ □ □                 ▦
    x, y = imageDims
    dataLen = ((x-2)*(y-2)*4 + (x-2+y-2)*5 + 6) * deformLen^2

    data = zeros(Float64, dataLen)
    index = Vector{NTuple{2,Ti}}(dataLen)
    idx = 0

    pixelRange = CartesianRange(imageDims)
    pixelFirst, pixelEnd = first(pixelRange), last(pixelRange)
    @inbounds for ii in pixelRange
        i = sub2ind(imageDims, ii[1], ii[2])
        neighborRange = CartesianRange(max(pixelFirst, ii-pixelFirst), min(pixelEnd, ii+pixelFirst))
        for jj in neighborRange
            if jj < ii
                j = sub2ind(imageDims, jj[1], jj[2])
                for a in eachindex(deformers), b in eachindex(deformers)
                    idx += 1
                    data[idx] = e^-truncated_quadratic_diff(deformers[a], deformers[b], rate, threshold)
                    index[idx] = ind2sub(𝐇²Dims, sub2ind(𝐇⁴Dims, i, a, j, b))
                end
            end
        end
    end
    return PSSTensor(weight*data, index, 𝐇²Dims)
end
