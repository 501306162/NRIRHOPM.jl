"""
    unaryclique(fixedImg, movingImg, labels)
    unaryclique(fixedImg, movingImg, labels, potential)
    unaryclique(fixedImg, movingImg, labels, potential, weight)

Returns the **data cost** of unary-cliques.
"""
function unaryclique{T,N}(fixedImg::Array{T,N}, movingImg::Array{T,N}, labels::Array{NTuple{N}}, potential::DataCost=SAD(), weight::Real=1)
    logger = get_logger(current_module())
    debug(logger, "Calling unaryclique with weight=$weight...")
    return weight*potential.𝓕(fixedImg, movingImg, labels)
end


"""
    pairwiseclique(imageDims, labels)
    pairwiseclique(imageDims, labels, potential)
    pairwiseclique(imageDims, labels, potential, weight)

Returns the **smooth cost** of pairwise-cliques.
"""
function pairwiseclique{N}(imageDims::NTuple{N}, labels::Array{NTuple{N}}, potential::SmoothCost=TAD(), weight::Real=1)
    logger = get_logger(current_module())
    debug(logger, "Calling pairwiseclique with weight=$weight...")
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum)
    labels = reshape(labels, labelNum)
    args = map(x->getfield(potential,x), fieldnames(potential)[2:end])
    block = [potential.𝓕(α, β, args...) for α in labels, β in labels]
    block = e.^-block
    return BSSTensor([TensorBlock(weight*block, neighbors(SquareCubic,imageDims), tensorDims)], tensorDims)
end


"""
    treyclique(imageDims, labels)
    treyclique(imageDims, labels, potential)
    treyclique(imageDims, labels, potential, weight)

Returns the **high order cost** of 3-element-cliques.
"""
function treyclique(imageDims::NTuple{2}, labels::Array{NTuple{2}}, potential::TopologyCost2D=TP2D(), weight::Real=1)
    logger = get_logger(current_module())
    debug(logger, "Calling treyclique with weight=$weight...")
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum, pixelNum, labelNum)
    labels = reshape(labels, labelNum)
    #   □ ⬓ □        ⬓                ⬓      r,c-->    ⬔ => p1 => α
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ⬓ => p2 => β
    #   □ ⬓ □              ⬓    ⬓            ↓         ▦ => p3 => χ
    #              Jᵇᵇ   Jᶠᵇ    Jᶠᶠ   Jᵇᶠ
    indexJᶠᶠ, indexJᵇᶠ, indexJᶠᵇ, indexJᵇᵇ = neighbors(Connected8{3}, imageDims)

    blockJᶠᶠ = [potential.Jᶠᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᶠ = [potential.Jᵇᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᶠᵇ = [potential.Jᶠᵇ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᵇ = [potential.Jᵇᵇ(α, β, χ) for α in labels, β in labels, χ in labels]

    return BSSTensor([TensorBlock(weight*blockJᶠᶠ, indexJᶠᶠ, tensorDims),
                      TensorBlock(weight*blockJᵇᶠ, indexJᵇᶠ, tensorDims),
                      TensorBlock(weight*blockJᶠᵇ, indexJᶠᵇ, tensorDims),
                      TensorBlock(weight*blockJᵇᵇ, indexJᵇᵇ, tensorDims)], tensorDims)
end


"""
    quadraclique(imageDims, labels)
    quadraclique(imageDims, labels, potential)
    quadraclique(imageDims, labels, potential, weight)

Returns the **high order cost** for 4-element-cliques.
"""
function quadraclique(imageDims::NTuple{3}, labels::Array{NTuple{3}}, potential::TopologyCost3D=TP3D(), weight::Real=1)
    logger = get_logger(current_module())
    debug(logger, "Calling quadraclique with weight=$weight...")
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum, pixelNum, labelNum, pixelNum, labelNum)
    labels = reshape(labels, labelNum)
    # coordinate system(r,c,z):
    #  up  r     c --->        z × × (front to back)
    #  to  |   left to right     × ×
    # down ↓
    # point => label:
    # p1 => α   p2 => β   p3 => χ   p5 => δ
    indexJᶠᶠᶠ, indexJᵇᶠᶠ, indexJᶠᵇᶠ, indexJᵇᵇᶠ, indexJᶠᶠᵇ, indexJᵇᶠᵇ, indexJᶠᵇᵇ, indexJᵇᵇᵇ = neighbors(Connected26{4}, imageDims)

    blockJᶠᶠᶠ = [potential.Jᶠᶠᶠ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᵇᶠᶠ = [potential.Jᵇᶠᶠ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᶠᵇᶠ = [potential.Jᶠᵇᶠ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᵇᵇᶠ = [potential.Jᵇᵇᶠ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᶠᶠᵇ = [potential.Jᶠᶠᵇ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᵇᶠᵇ = [potential.Jᵇᶠᵇ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᶠᵇᵇ = [potential.Jᶠᵇᵇ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]
    blockJᵇᵇᵇ = [potential.Jᵇᵇᵇ(α, β, χ, δ) for α in labels, β in labels, χ in labels, δ in labels]

    return BSSTensor([TensorBlock(weight*blockJᶠᶠᶠ, indexJᶠᶠᶠ, tensorDims),
                      TensorBlock(weight*blockJᵇᶠᶠ, indexJᵇᶠᶠ, tensorDims),
                      TensorBlock(weight*blockJᶠᵇᶠ, indexJᶠᵇᶠ, tensorDims),
                      TensorBlock(weight*blockJᵇᵇᶠ, indexJᵇᵇᶠ, tensorDims),
                      TensorBlock(weight*blockJᶠᶠᵇ, indexJᶠᶠᵇ, tensorDims),
                      TensorBlock(weight*blockJᵇᶠᵇ, indexJᵇᶠᵇ, tensorDims),
                      TensorBlock(weight*blockJᶠᵇᵇ, indexJᶠᵇᵇ, tensorDims),
                      TensorBlock(weight*blockJᵇᵇᵇ, indexJᵇᵇᵇ, tensorDims)], tensorDims)
end
