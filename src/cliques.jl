"""
    unaryclique(fixedImg, movingImg, labels)
    unaryclique(fixedImg, movingImg, labels, potential)

Construct the **data cost**.
"""
function unaryclique{T,N,P<:DataCost}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}}, potential::P=SAD())
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Fixed image and moving image are not in the same size!"))
    info("Calling unaryclique($P): ")
    return potential.𝓕(fixedImg, movingImg, labels)
end


"""
    pairwiseclique(fixedImg, movingImg, labels, weight)
    pairwiseclique(fixedImg, movingImg, labels, weight, potential)
    pairwiseclique(imageDims, labels, potential)
    pairwiseclique(imageDims, labels, potential, weight)

Construct the **smooth cost**.
"""
function pairwiseclique{T,N}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}}, weight::Real, potential::SmoothTerm=TAD())
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Fixed image and moving image are not in the same size!"))
    pairwiseclique(imageDims, reshape(labels, length(labels)), potential, weight)
end

function pairwiseclique{N,P<:SmoothCost}(imageDims::NTuple{N}, labels::Vector{NTuple{N}}, potential::P, weight=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum)
    info("Calling pairwiseclique($P) with weight=$weight: ")
    args = map(x->getfield(potential,x), fieldnames(potential)[2:end])
    block = [potential.𝓕(α, β, args...) for α in labels, β in labels]
    block = e.^-block
    return BSSTensor([TensorBlock(weight*block, neighbors(Connected8{2},imageDims), tensorDims)], tensorDims)
end


"""
    treyclique(fixedImg, movingImg, labels, weight)
    treyclique(fixedImg, movingImg, labels, weight, potential)
    treyclique(imageDims, labels, potential)
    treyclique(imageDims, labels, potential, weight)

Construct the **high order cost** for topology preserving.
"""
function treyclique{T,N}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}}, weight::Real, potential::TopologyCost=TP())
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Fixed image and moving image are not in the same size!"))
    treyclique(imageDims, reshape(labels, length(labels)), potential, weight)
end

function treyclique(imageDims::NTuple{2}, labels::Vector{NTuple{2}}, potential::TP, weight=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum, pixelNum, labelNum)
    info("Calling treyclique(Topology Preserving) with weight=$weight: ")
    #   □ ⬓ □        ⬓                ⬓      y,x-->    ⬔ => ii => p1
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ▦ => jj => p2
    #   □ ⬓ □              ⬓    ⬓            ↓         ⬓ => kk => p3
    #              Jᵇᵇ   Jᵇᶠ    Jᶠᶠ   Jᶠᵇ
    indexJᶠᶠ, indexJᵇᶠ, indexJᶠᵇ, indexJᵇᵇ = neighbors(Connected8{3},imageDims)

    blockJᶠᶠ = [potential.Jᶠᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᶠ = [potential.Jᵇᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᶠᵇ = [potential.Jᶠᵇ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᵇ = [potential.Jᵇᵇ(α, β, χ) for α in labels, β in labels, χ in labels]

    return BSSTensor([TensorBlock(weight*e.^-blockJᶠᶠ, indexJᶠᶠ, tensorDims),
                      TensorBlock(weight*e.^-blockJᵇᶠ, indexJᵇᶠ, tensorDims),
                      TensorBlock(weight*e.^-blockJᶠᵇ, indexJᶠᵇ, tensorDims),
                      TensorBlock(weight*e.^-blockJᵇᵇ, indexJᵇᵇ, tensorDims)], tensorDims)
end
