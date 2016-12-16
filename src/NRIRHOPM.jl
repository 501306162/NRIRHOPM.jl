module NRIRHOPM
using Reexport
using Interpolations
@reexport using Plots

import Base: ==

export TensorBlock, BSSTensor, SSTensor, ⊙, hopm
export Connected4, Connected8, Connected6, Connected26, neighbors
export AbstractPotential,
       UnaryPotential, DataTerm, DataCost,
       PairwisePotential, SmoothTerm, SmoothCost, RegularTerm,
       TreyPotential, TopologyCost
export SAD, SSD,
       Potts, TAD, TQD,
       TP
export unaryclique, pairwiseclique, treyclique
export meshgrid
export dirhop, registering

include("tensors.jl")
include("hopm.jl")
include("neighbors.jl")
include("types.jl")
include("potentials.jl")
include("cliques.jl")
include("utils.jl")

function dirhop(fixedImg, movingImg, labels; datacost::DataCost=SAD(),
                smooth::SmoothCost=TAD(), trey::TopologyCost=TP(),
                α::Real=1,                β::Real=1)

    pixelNum = length(fixedImg)
    labelNum = length(labels)

    @time 𝐇¹ = unaryclique(fixedImg, movingImg, labels; algorithm=datacost)
	@time 𝐇² = pairwiseclique(fixedImg, movingImg, deformableWindow)
    if β == 0
        @time score, 𝐯 = hopm(𝐇¹, 𝐇²)
    else
        @time 𝐇³ = treyclique(fixedImg, movingImg, deformableWindow; algorithm=trey, ω=γ)
        @time score, 𝐯 = hopm(𝐇¹, 𝐇², 𝐇³)
    end
    𝐌 = reshape(𝐯, pixelNum, labelNum)
    return score, [findmax(𝐌[i,:])[2] for i in 1:pixelNum], 𝐌
end

function registering(movingImg, labels, indicator::Vector{Int})
    imageDims = size(movingImg)
    registeredImg = similar(movingImg)
    quivers = Matrix(imageDims)
    for 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        quivers[𝒊] = labels[indicator[i]]
        𝐭 = CartesianIndex(quivers[𝒊])
        registeredImg[𝒊] = movingImg[𝒊+𝐭]
    end
    return registeredImg, quivers
end

end # module
