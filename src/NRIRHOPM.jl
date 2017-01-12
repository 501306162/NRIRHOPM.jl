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
export unaryclique, pairwiseclique, treyclique, quadraclique
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
                smooth::SmoothCost=TAD(), topology::TopologyCost=TP(),
                α::Real=1,                β::Real=1,
                hopmtol=1e-5, hopmMaxIter=300, verbose::Bool=true)
    imageDims = size(fixedImg)
    imageDims == size(movingImg) || throw(ArgumentError("Fixed image and moving image are not in the same size!"))
    pixelNum = length(fixedImg)
    labelNum = length(labels)

    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐇¹ = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
	@time 𝐇² = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    𝐯₀ = rand(length(𝐇¹))

    if β == 0
        @time score, 𝐯 = hopm(𝐇¹, 𝐇², 𝐯₀, hopmtol, hopmMaxIter, verbose)
    elseif length(imageDims) == 2
        verbose && info("Calling treyclique(Topology-Preserving-2D) with weight=$β: ")
        @time 𝐇³ = treyclique(fixedImg, movingImg, labels, β, topology)
        @time score, 𝐯 = hopm(𝐇¹, 𝐇², 𝐇³, 𝐯₀, hopmtol, hopmMaxIter, verbose)
    elseif length(imageDims) == 3
        info("Calling quadraclique(Topology-Preserving-3D) with weight=$β: ")
        @time 𝐇⁴ = quadraclique(fixedImg, movingImg, labels, β, topology)
        @time score, 𝐯 = hopm(𝐇¹, 𝐇², 𝐇⁴, 𝐯₀, hopmtol, hopmMaxIter, verbose)
    end
    𝐌 = reshape(𝐯, pixelNum, labelNum)
    return score, [findmax(𝐌[i,:])[2] for i in 1:pixelNum], 𝐌
end

function registering(movingImg, labels, indicator::Vector{Int})
    imageDims = size(movingImg)
    registeredImg = similar(movingImg)
    quivers = Array{Any,length(imageDims)}(imageDims...)
    for 𝒊 in CartesianRange(imageDims)
        i = sub2ind(imageDims, 𝒊.I...)
        quivers[𝒊] = labels[indicator[i]]
        𝐭 = CartesianIndex(quivers[𝒊])
        registeredImg[𝒊] = movingImg[𝒊+𝐭]
    end
    return registeredImg, quivers
end

end # module
