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


function warp{N,T<:Real,Dim}(movingImg, displacement::Array{Vec{N,T},Dim})
    imageDims = size(movingImg)
    gridDims = size(displacement)
    warppedImg = zeros(imageDims)
    if imageDims != gridDims
        knots = ntuple(x->linspace(1, imageDims[x], gridDims[x]), Val{N})
        displacementITP = interpolate(knots, displacement, Gridded(Linear()))
        movingImgITP = interpolate(movingImg, BSpline(Linear()), OnGrid())
        for 𝒊 in CartesianRange(imageDims)
            𝐭 = Vec(𝒊.I...) + displacementITP[𝒊]
            warppedImg[𝒊] = movingImgITP[𝐭...]
        end
    else
        for 𝒊 in CartesianRange(imageDims)
            𝐭 = 𝒊 + CartesianIndex(displacement[𝒊]...)
            if checkbounds(Bool, movingImg, 𝐭)
                warppedImg[𝒊] = movingImg[𝐭]
            else
                warn("𝐭($𝐭) is outbound, skipped.")
            end
        end
    end
    return warppedImg
end

function upsample{N}(gridDimsUp::NTuple{N}, gridDims::NTuple{N}, spectrum::Matrix)
    spectrumVec = reshape([Vec(spectrum[i,:]) for i = 1:prod(gridDims)], gridDims)
    knots = ntuple(x->linspace(1, gridDimsUp[x], gridDims[x]), Val{N})
    spectrumVecITP = interpolate(knots, spectrumVec, Gridded(Linear()))

    spectrumInterpolated = zeros(prod(gridDimsUp), size(spectrum,2))
    for 𝒊 in CartesianRange(gridDimsUp)
        r = sub2ind(gridDimsUp, 𝒊.I...)
        spectrumInterpolated[r,:] = collect(spectrumVecITP[𝒊])
    end

    return spectrumInterpolated
end


function multilevel(fixedImg, movingImg, labelSets::Vector, grids::Vector{NTuple},
                    datacost::DataCost=SAD(), smooth::SmoothCost=TAD(), topology::TopologyCost=TP(),
                                              α::Real=1,                β::Real=1; hopmkwargs...)
    # init
    level = length(labelRanges)
    fixedImgs = Vector(level)
    movingImgs = Vector(level)
    displacements = Vector(level)
    spectrums = Vector(level)

    # topology preservation pre-processing
    fixedImgs[1] = copy(fixedImg)
    movingImgs[1] = copy(movingImg)
    gridDims = grids[1]
    labels = labelSet[1]
    energy[1], spectrums[1] = optimize(fixedImgs[1], movingImgs[1], gridDims, labels, datacost, smooth, topology, α, β, hopmkwargs...)

    # loop
    for l = 2:level
        # upsample spectrum to latest level
        spectrumSampled = upsample(grids[l], gridDims, spectrums[l-1])
        labels = labelSets[l]
        energy, spectrum = optimize(fixedImgs[l], movingImgs[l], grids[l], labels, datacost, smooth, α; hopmkwargs..., 𝐒₀=spectrumSampled)
        spectrums[l] = spectrum
        indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
        displacements[l] = reshape([Vec(labels[i]) for i in indicator], grids[l])
        movingImgs[l+1] = warp(movingImgs[l], displacement[l])
    end

    return fixedImgs, movingImgs, displacements, spectrums
end
