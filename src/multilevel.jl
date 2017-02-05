function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                       imageDims::NTuple{N}, labels::Array{NTuple{N}},
                       datacost::DataCost, α::Real,
                       smooth::SmoothCost, β::Real;
                       𝐒₀::Matrix=rand(length(fixedImg),length(labels)),
                       tolerance::Float64=1e-5,
                       maxIteration::Integer=300,
                       constrainRow::Bool=false
                      )
    logger = get_logger(current_module())
    info(logger, "Timing unaryclique with weight=$α: ")
    @timelog dcost = unaryclique(fixedImg, movingImg, labels, datacost, α)

    if size(fixedImg) == imageDims
        𝐡 = reshape(dcost, length(dcost))
    else
        dcostDownsampled = upsample(imageDims, size(fixedImg), dcost)   # this is actually downsampling
        𝐡 = reshape(dcostDownsampled, length(dcostDownsampled))
    end

    info(logger, "Timing pairwiseclique with weight=$β: ")
    @timelog 𝐇 = pairwiseclique(imageDims, labels, smooth, β)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    info(logger, "Timing HOPM constrained by $(constrainRow ? "row" : "vecnorm"): ")
    @timelog energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, constrainRow)

    return energy, spectrum
end

function optimize{T}(fixedImg::AbstractArray{T,2}, movingImg::AbstractArray{T,2},
                     imageDims::NTuple{2}, labels::Array{NTuple{2}},
                     datacost::DataCost, α::Real,
                     smooth::SmoothCost, β::Real,
                     topology::TopologyCost2D, χ::Real;
                     𝐒₀::Matrix=rand(length(fixedImg),length(labels)), tolerance::Float64=1e-5,
                     maxIteration::Integer=300, constrainRow::Bool=false)
    logger = get_logger(current_module())
    info(logger, "Timing unaryclique with weight=$α: ")
    @timelog dcost = unaryclique(fixedImg, movingImg, labels, datacost, α)
    if size(fixedImg) == imageDims
        𝐡 = reshape(dcost, length(dcost))
    else
        dcostDownsampled = upsample(imageDims, size(fixedImg), dcost)   # this is actually downsampling
        𝐡 = reshape(dcostDownsampled, length(dcostDownsampled))
    end

    info(logger, "Timing pairwiseclique with weight=$β: ")
    @timelog 𝐇 = pairwiseclique(imageDims, labels, smooth, β)

    info(logger, "Timing treyclique with weight=$χ: ")
    @timelog 𝑯 = treyclique(imageDims, labels, topology, χ)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    info(logger, "Timing HOPM constrained by $(constrainRow ? "row" : "vecnorm"): ")
    @timelog energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝐒₀, tolerance, maxIteration, constrainRow)

    return energy, spectrum
end


function optimize{T}(fixedImg::AbstractArray{T,3}, movingImg::AbstractArray{T,3},
                     imageDims::NTuple{3}, labels::Array{NTuple{3}},
                     datacost::DataCost, α::Real,
                     smooth::SmoothCost, β::Real,
                     topology::TopologyCost3D, χ::Real;
                     𝐒₀::Matrix=rand(length(fixedImg),length(labels)), tolerance::Float64=1e-5,
                     maxIteration::Integer=300, constrainRow::Bool=false)
    logger = get_logger(current_module())
    info(logger, "Timing unaryclique with weight=$α: ")
    @timelog dcost = unaryclique(fixedImg, movingImg, labels, datacost, α)
    if size(fixedImg) == imageDims
        𝐡 = reshape(dcost, length(dcost))
    else
        dcostDownsampled = upsample(imageDims, size(fixedImg), dcost)   # this is actually downsampling
        𝐡 = reshape(dcostDownsampled, length(dcostDownsampled))
    end

    info(logger, "Timing pairwiseclique with weight=$β: ")
    @timelog 𝐇 = pairwiseclique(imageDims, labels, smooth, β)

    info(logger, "Timing quadraclique with weight=$χ: ")
    @timelog 𝑯 = quadraclique(imageDims, labels, topology, χ)

    if eltype(𝐡) != eltype(𝐒₀)
        𝐒₀ = convert(Matrix{eltype(𝐡)}, 𝐒₀)
    end

    info(logger, "Timing HOPM constrained by $(constrainRow ? "row" : "vecnorm"): ")
    @timelog energy, spectrum = hopm_mixed(𝐡, 𝐇, 𝑯, 𝐒₀, tolerance, maxIteration, constrainRow)

    return energy, spectrum
end


function warp{N,T<:Real,Dim}(movingImg, displacement::Array{Vec{N,T},Dim})
    logger = get_logger(current_module())
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
                warn(logger, "𝐭($𝐭) is outbound, skipped.")
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

function multilevel(fixedImg, movingImg, labelSets, grids;
                    datacost::DataCost=SAD(), α::Real=1,
                    smooth::SmoothCost=TAD(), β::Real=1,
                    topology::TopologyCost3D=TP3D(), χ::Real=1,
                    tolerance::Float64=1e-5,
                    maxIteration::Integer=300,
                    constrainRow::Bool=false
                   )
    logger = get_logger(current_module())
    info(logger, "Start multilevel processing...")
    level = length(labelSets)
    movingImgs = Vector(level)
    displacements = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)

    gridDims = grids[1]
    labels = labelSets[1]
    info(logger, "Level 1:")
    info(logger, "Image Dimension: $(size(fixedImg))")
    info(logger, "Grid Dimension: $(gridDims)")
    info(logger, "Label Total Number: $(length(labels))")
    𝐒₀ = rand(prod(gridDims), length(labels))
    energy[1], spectrum = optimize(fixedImg, movingImg, gridDims, labels,
                                   datacost, α, smooth, β, topology, χ,
                                   𝐒₀=𝐒₀, tolerance=tolerance,
                                   maxIteration=maxIteration, constrainRow=constrainRow)
    spectrums[1] = spectrum
    indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
    displacements[1] = reshape([Vec(labels[i]) for i in indicator], grids[1])
    movingImgs[1] = warp(movingImg, displacements[1])

    for l = 2:level
        labels = labelSets[l]
        info(logger, "Level $l: ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(grids[l])")
        info(logger, "Label Total Number: $(length(labels))")
        # upsample spectrum to latest level
        spectrumSampled = upsample(grids[l], gridDims, spectrums[l-1])
        energy, spectrum = optimize(fixedImg, movingImgs[l-1], grids[l], labels,
                                    datacost, α, smooth, β,
                                    𝐒₀=spectrumSampled, tolerance=tolerance,
                                    maxIteration=maxIteration, constrainRow=constrainRow)
        spectrums[l] = spectrum
        indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
        displacements[l] = reshape([Vec(labels[i]) for i in indicator], grids[l])
        movingImgs[l] = warp(movingImgs[l-1], displacements[l])
    end
    return movingImgs, displacements, spectrums
end
