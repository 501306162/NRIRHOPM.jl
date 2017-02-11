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
    displacementFields = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)

    gridDims = grids[1]
    labels = labelSets[1]
    info(logger, "Level 1:")
    info(logger, "Image Dimension: $(size(fixedImg))")
    info(logger, "Grid Dimension: $(gridDims)")
    info(logger, "Label Total Number: $(length(labels))")
    𝐒₀ = rand(prod(gridDims), length(labels))
    energy[1], spectrum = optimize(fixedImg, movingImg, displacements, gridDims,
                                   datacost, α, smooth, β, topology, χ,
                                   MixHOPM(), spectrum)
    spectrums[1] = spectrum
    indicator = [indmax(spectrum[:,i]) for i in indices(spectrum,2)]
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
        indicator = [indmax(spectrum[:,i]) for i in indices(spectrum,2)]
        displacementFields[l] = fieldlize(indicator, labelSets[l], grids[l])
        movingImgs[l] = warp(movingImgs[l-1], displacements[l])
    end
    return movingImgs, displacements, spectrums
end
