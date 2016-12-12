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
    pairwiseclique(imageDims, labels, potential, weight)
end

function pairwiseclique{N,P<:SmoothCost}(imageDims::NTuple{N}, labels::Array{NTuple{N}}, potential::P, weight=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    info("Calling pairwiseclique($P): ")
    args = map(x->getfield(potential,x), fieldnames(potential)[2:end])
    block = [potential.𝓕(α, β, args...) for α in labels, β in labels]
    block = e.^-block
    return BSSTensor(weight*block, neighbors(Connected8{2},imageDims), (pixelNum, labelNum, pixelNum, labelNum))
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
    treyclique(imageDims, labels, potential, weight)
end

function treyclique{N,P<:TopologyCost}(imageDims::NTuple{N}, labels::Array{NTuple{N}}, potential::P, weight=1)
    pixelNum = prod(imageDims)
    labelNum = length(labels)
    info("Calling treyclique($P): ")
    # for a in eachindex(deformers), b in eachindex(deformers), c in eachindex(deformers)
    #     indexNum += 1
    #     cost = topology_preserving([jj.I...], [ii.I...], [kk.I...], deformers[a], deformers[b], deformers[c])
    #     data[indexNum] = -cost
    # end
    return BSSTensor(weight*block, neighbors(Connected8{3},imageDims), (pixelNum, labelNum, pixelNum, labelNum, pixelNum, labelNum))
end
