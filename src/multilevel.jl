function multilevel(fixedImg, movingImg, labels, datacost::DataCost=SAD(),
                    smooth::SmoothCost=TAD(), topology::TopologyCost=TP(),
                    α::Real=1,                β::Real=1; hopmkwargs...)
    # init


    # loop
    for level in levels
        energy, spectrum = optimize(fixedGrid, movingGrid, labels, datacost, smooth, α, hopmkwargs...)
        spectrumNew = upsample(spectrum)
        movingGridNew = register(movingGrid, labels, quivers)
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

function upsample{N}(moving, imageDims::NTuple{N}, displacementField)
    registeredImg = zeros(imageDims)
    knots = ntuple(x->1:imageDims[x], Val{N})
    itp = interpolate(knots, displacementField, Gridded(Linear()))
end

function register(movingImg, labels, displacement)
    registeredImg = similar(movingImg)
    for 𝒊 in CartesianRange(size(movingImg))
        𝐝 = 𝒊 + CartesianIndex(displacement[𝒊])
        if checkbounds(Bool, movingImg, 𝐝)
            warn("𝐝($𝐝) is outbounds, skipped.")
        else
            registeredImg[𝒊] = movingImg[𝐝]
        end
    end
    return registeredImg
end
