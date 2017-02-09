"""
    unaryclique(fixedImg, movingImg, displacements)
    unaryclique(fixedImg, movingImg, displacements, model)
    unaryclique(fixedImg, movingImg, displacements, model, weight)

Returns the **data cost** of unary-cliques.
"""
@generated function unaryclique{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                          displacements::AbstractArray{NTuple{N}}, model::DataCost=SAD(), weight::Real=1)
    args = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = pop!(args)
    ret = :(weight * $func(fixedImg, movingImg, displacements))
end


"""
    pairwiseclique(imageDims, labels)
    pairwiseclique(imageDims, labels, potential)
    pairwiseclique(imageDims, labels, potential, weight)

Returns the **smooth cost** of pairwise-cliques.
"""
function pairwiseclique{N}(imageDims::NTuple{N}, labels::AbstractArray{NTuple{N}}, potential::SmoothCost=TAD(), weight::Real=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = (pixelNum, labelNum, pixelNum, labelNum)
    labels = reshape(labels, labelNum)
    args = map(x->getfield(potential,x), fieldnames(potential)[2:end])
    block = [potential.f(α, β, args...) for α in labels, β in labels]
    block = e.^-block
    return BlockedTensor([ValueBlock(weight*block)], [IndexBlock(neighbors(SquareCubic,imageDims))], tensorDims)
end


"""
    treyclique(imageDims, displacements)
    treyclique(imageDims, displacements, model)
    treyclique(imageDims, displacements, model, weight)

Returns the **high order cost** of 3-element-cliques.
"""
function treyclique(imageDims::NTuple{2}, displacements::AbstractArray{NTuple{2}}, model::TP2D=TP2D(), weight::Real=1)
    displacements = reshape(labels, labelNum)
    #   □ ⬓ □        ⬓                ⬓      r,c-->    ⬔ => p1 => α
    #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ⬓ => p2 => β
    #   □ ⬓ □              ⬓    ⬓            ↓         ▦ => p3 => χ
    #              Jᵇᵇ   Jᶠᵇ    Jᶠᶠ   Jᵇᶠ
    indexJᶠᶠ, indexJᵇᶠ, indexJᶠᵇ, indexJᵇᵇ = neighbors(Connected8{3}, imageDims)

    blockJᶠᶠ = [model.Jᶠᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᶠ = [model.Jᵇᶠ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᶠᵇ = [model.Jᶠᵇ(α, β, χ) for α in labels, β in labels, χ in labels]
    blockJᵇᵇ = [model.Jᵇᵇ(α, β, χ) for α in labels, β in labels, χ in labels]

    return BlockedTensor([ValueBlock(weight*blockJᶠᶠ),ValueBlock(weight*blockJᵇᶠ),ValueBlock(weight*blockJᶠᵇ),ValueBlock(weight*blockJᵇᵇ)],
                         [IndexBlock(indexJᶠᶠ),       IndexBlock(indexJᵇᶠ),       IndexBlock(indexJᶠᵇ),       IndexBlock(indexJᵇᵇ)],
                         ntuple(x -> isodd(x) ? length(labels) : prod(imageDims), 6))
end


"""
    quadraclique(imageDims, labels)
    quadraclique(imageDims, labels, potential)
    quadraclique(imageDims, labels, potential, weight)

Returns the **high order cost** for 4-element-cliques.
"""
function quadraclique{T}(imageDims::NTuple{3,Int}, displacements::AbstractArray{NTuple{3,T}}, potential::TP3D=TP3D(), weight::Real=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    tensorDims = ntuple(x -> isodd(x) ? labelNum : pixelNum, 6)
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

    return BlockedTensor([TensorBlock(weight*blockJᶠᶠᶠ, indexJᶠᶠᶠ, tensorDims),
                      TensorBlock(weight*blockJᵇᶠᶠ, indexJᵇᶠᶠ, tensorDims),
                      TensorBlock(weight*blockJᶠᵇᶠ, indexJᶠᵇᶠ, tensorDims),
                      TensorBlock(weight*blockJᵇᵇᶠ, indexJᵇᵇᶠ, tensorDims),
                      TensorBlock(weight*blockJᶠᶠᵇ, indexJᶠᶠᵇ, tensorDims),
                      TensorBlock(weight*blockJᵇᶠᵇ, indexJᵇᶠᵇ, tensorDims),
                      TensorBlock(weight*blockJᶠᵇᵇ, indexJᶠᵇᵇ, tensorDims),
                      TensorBlock(weight*blockJᵇᵇᵇ, indexJᵇᵇᵇ, tensorDims)], tensorDims)
end
