
# function multilevel(fixedImg, movingImg, labels, datacost::DataCost=SAD(),
#                     smooth::SmoothCost=TAD(), topology::TopologyCost=TP(),
#                     α::Real=1,                β::Real=1,
#                     hopmtol=1e-5, hopmMaxIter=300, verbose::Bool=true)
#     for level in levels
#         energy, spectrum = optimize(fixedGrid, movingGrid, datacost, smooth, α, spectrum, tolerance, maxIteration, verbose)
#         indicator = intergerlize()
#         registered, quivers = register(indicator)
#     end
# end

function optimize{N}(fixedImg::AbstractArray{N}, movingImg::AbstractArray{N},
                     datacost::DataCost, smooth::SmoothCost, topology::TopologyCost,
                                         α::Real,            β::Real,
                     𝐒₀::Matrix, tolerance::Float64, maxIteration::Integer, verbose::Bool)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    @time energy, spectrum = hopm(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, verbose)

    return energy, spectrum
end

function optimize(fixedImg::AbstractArray{2}, movingImg::AbstractArray{2},
                  datacost::DataCost, smooth::SmoothCost, topology::TopologyCost,
                                      α::Real,            β::Real,
                  𝐒₀::Matrix, tolerance::Float64, maxIteration::Integer, verbose::Bool)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    verbose && info("Calling treyclique(Topology-Preserving-2D) with weight=$β: ")
    @time 𝑯 = treyclique(fixedImg, movingImg, labels, β, topology)
    @time energy, spectrum = hopm(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, verbose)

    return energy, spectrum
end


function optimize(fixedImg::AbstractArray{3}, movingImg::AbstractArray{3},
                  datacost::DataCost, smooth::SmoothCost, topology::TopologyCost,
                                      α::Real,            β::Real,
                  𝐒₀::Matrix, tolerance::Float64, maxIteration::Integer, verbose::Bool)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    verbose && info("Calling quadraclique(Topology-Preserving-3D) with weight=$β: ")
    @time 𝑯 = quadraclique(fixedImg, movingImg, labels, β, topology)
    @time energy, spectrum = hopm(𝐡, 𝐇, 𝑯, 𝐒₀, tolerance, maxIteration, verbose)

    return energy, spectrum
end

# function register(imageDims, gridDims, level, indicator)
#
# end



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
        @time energy, 𝐯 = hopm_canonical(𝐇¹, 𝐇², 𝐯₀, hopmtol, hopmMaxIter, verbose)
    elseif length(imageDims) == 2
        verbose && info("Calling treyclique(Topology-Preserving-2D) with weight=$β: ")
        @time 𝐇³ = treyclique(fixedImg, movingImg, labels, β, topology)
        @time energy, 𝐯 = hopm_canonical(𝐇¹, 𝐇², 𝐇³, 𝐯₀, hopmtol, hopmMaxIter, verbose)
    elseif length(imageDims) == 3
        verbose && info("Calling quadraclique(Topology-Preserving-3D) with weight=$β: ")
        @time 𝐇⁴ = quadraclique(fixedImg, movingImg, labels, β, topology)
        @time energy, 𝐯 = hopm_canonical(𝐇¹, 𝐇², 𝐇⁴, 𝐯₀, hopmtol, hopmMaxIter, verbose)
    end
    𝐌 = reshape(𝐯, pixelNum, labelNum)
    return energy, [findmax(𝐌[i,:])[2] for i in 1:pixelNum], 𝐌
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
