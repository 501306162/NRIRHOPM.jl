function multilevel(fixedImg, movingImg, labels, datacost::DataCost=SAD(),
                    smooth::SmoothCost=TAD(), topology::TopologyCost=TP(),
                    α::Real=1,                β::Real=1; hopmkwargs...)
    for level in levels
        energy, spectrum = optimize(fixedGrid, movingGrid, labels, datacost, smooth, α, hopmkwargs...)

        # registered, quivers = register(indicator)
        # spectrum_new = interpolation(spectrum)
    end
end

function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, labels::Array{NTuple{N}},
                       datacost::DataCost, smooth::SmoothCost, α::Real;
                       𝐒₀::Matrix=rand(length(fixedImg),length(labels)), tolerance::Float64=1e-5,
                       maxIteration::Integer=300, constrainRow::Bool=false, verbose::Bool=false)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    @time energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, constrainRow, verbose)

    return energy, spectrum
end

function optimize{T}(fixedImg::Array{T,2}, movingImg::Array{T,2}, labels::Array{NTuple{2}},
                     datacost::DataCost, smooth::SmoothCost, topology::TopologyCost,
                                         α::Real,            β::Real;
                     𝐒₀::Matrix=rand(length(fixedImg),length(labels)), tolerance::Float64=1e-5,
                     maxIteration::Integer=300, constrainRow::Bool=false, verbose::Bool=false)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    verbose && info("Calling treyclique(Topology-Preserving-2D) with weight=$β: ")
    @time 𝑯 = treyclique(fixedImg, movingImg, labels, β, topology)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    @time energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, constrainRow, verbose)

    return energy, spectrum
end


function optimize{T}(fixedImg::AbstractArray{T,3}, movingImg::AbstractArray{T,3}, labels::Array{NTuple{3}},
                  datacost::DataCost, smooth::SmoothCost, topology::TopologyCost,
                                      α::Real,            β::Real;
                  𝐒₀::Matrix=rand(length(fixedImg),length(labels)), tolerance::Float64=1e-5,
                  maxIteration::Integer=300, constrainRow::Bool=false, verbose::Bool=false)
    verbose && info("Calling unaryclique($datacost): ")
    @time 𝐡 = unaryclique(fixedImg, movingImg, labels, datacost)

    verbose && info("Calling pairwiseclique($smooth) with weight=$α: ")
    @time 𝐇 = pairwiseclique(fixedImg, movingImg, labels, α, smooth)

    verbose && info("Calling quadraclique(Topology-Preserving-3D) with weight=$β: ")
    @time 𝑯 = quadraclique(fixedImg, movingImg, labels, β, topology)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    @time energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝑯, 𝐒₀, tolerance, maxIteration, constrainRow, verbose)

    return energy, spectrum
end

# function register(imageDims, gridDims, level, indicator)
#
# end

# function upsample(imageDims, gridDims, level, indicator)
#
# end

# function registering(movingImg, labels, indicator::Vector{Int})
#     imageDims = size(movingImg)
#     registeredImg = similar(movingImg)
#     quivers = Array{Any,length(imageDims)}(imageDims...)
#     for 𝒊 in CartesianRange(imageDims)
#         i = sub2ind(imageDims, 𝒊.I...)
#         quivers[𝒊] = labels[indicator[i]]
#         𝐭 = CartesianIndex(quivers[𝒊])
#         registeredImg[𝒊] = movingImg[𝒊+𝐭]
#     end
#     return registeredImg, quivers
# end
