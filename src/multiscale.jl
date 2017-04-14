function multilevel(fixedImg, movingImg, displacementSet, quantisations,
                    gridSet, method::AbstractHOPMMethod,
                    datacost::DataCost, αSet,
                    smooth::SmoothCost, βSet
                   )
    fixedImg = convert(AbstractArray{Float64}, fixedImg)
    movingImg = convert(AbstractArray{Float64}, movingImg)
    imageDims = size(fixedImg)
    level = length(displacementSet)
    warppedImgs = Vector(level+1)
    spectrums = Vector(level)
    energy = Vector(level)
    # displacementField = ndims(fixedImg) == 2 ? zeros(DVec2D, gridSet[1]) : zeros(DVec3D, gridSet[1])
    displacementFields = Vector(level)

    logger = get_logger(current_module())
    info(logger, "Start multilevel processing...")
    warppedImgs[1] = copy(movingImg)
    for l = 1:level
        info(logger, "Level $(l): ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridSet[l])")
        energy[l], spectrums[l] = optimize(fixedImg, warppedImgs[l], displacementSet[l],
                                           quantisations[l], gridSet[l], method,
                                           datacost, αSet[l], smooth, βSet[l])
        indicator = [indmax(spectrums[l][:,i]) for i in indices(spectrums[l],2)]
        # newField = fieldlize(indicator, displacementSet[l] .* quantisations[l], gridSet[l])
        # @time displacementField = fieldmerge([upsample(displacementField, gridSet[l]), newField])
        # warppedImgs[l+1] = warp(movingImg, upsample(displacementField, imageDims))
        displacementFields[l] = upsample(fieldlize(indicator, displacementSet[l], gridSet[l]), size(fixedImg))
        warppedImgs[l+1] = warp(warppedImgs[l], displacementFields[l])
    end
    info(logger, "Multilevel processing done!")
    return warppedImgs, displacementFields, spectrums, energy
end

function multilevel(fixedImg, movingImg, displacementSet, quantisations,
                    gridSet, method::AbstractHOPMMethod,
                    datacost::DataCost, αSet,
                    smooth::SmoothCost, βSet,
                    topology::TopologyCost, χSet
                   )
    fixedImg = convert(AbstractArray{Float64}, fixedImg)
    movingImg = convert(AbstractArray{Float64}, movingImg)
    level = length(displacementSet)
    warppedImgs = Vector(level+1)
    displacementFields = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)

    logger = get_logger(current_module())
    info(logger, "Start multilevel processing...")
    warppedImgs[1] = copy(movingImg)
    for l = 1:level
        info(logger, "Level $(l): ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridSet[l])")
        energy[l], spectrums[l] = optimize(fixedImg, warppedImgs[l], displacementSet[l],
                                           quantisations[l], gridSet[l], method,
                                           datacost, αSet[l], smooth, βSet[l], topology, χSet[l])
        indicator = [indmax(spectrums[l][:,i]) for i in indices(spectrums[l],2)]
        displacementFields[l] = upsample(fieldlize(indicator, displacementSet[l], gridSet[l]), size(fixedImg))
        warppedImgs[l+1] = warp(warppedImgs[l], displacementFields[l])
    end
    info(logger, "Multilevel processing done!")
    return warppedImgs, displacementFields, spectrums, energy
end


function multiresolution(fixedImg, movingImg, displacementSet, quantisations, method,
                         datacost::DataCost, αSet,
                         smooth::SmoothCost, βSet,
                         downsample::Integer=2,
                        )
    level = length(displacementSet)
    warppedImgs = Vector(level+1)
    displacementFields = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)
    fixedPyramid = gaussian_pyramid(fixedImg, level-1, downsample, 1)

    logger = get_logger(current_module())
    info(logger, "Start multiresolution processing...")
    warppedImgs[1] = copy(movingImg)
    for l = 1:level
        gridDims = size(fixedPyramid[1+level-l])
        warppedPyramid = gaussian_pyramid(warppedImgs[l], level-l, downsample, 1)
        info(logger, "Level $(l): ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridDims)")
        energy[l], spectrums[l] = optimize(fixedPyramid[1+level-l], warppedPyramid[end],
                                           displacementSet[l], quantisations[l],
                                           gridDims, method, datacost, αSet[l], smooth, βSet[l])
        indicator = [indmax(spectrums[l][:,i]) for i in indices(spectrums[l],2)]
        displacementFields[l] = upsample(fieldlize(indicator, displacementSet[l], gridDims), size(fixedImg))
        warppedImgs[l+1] = warp(warppedImgs[l], displacementFields[l])
    end
    info(logger, "Multiresolution processing done!")
    return warppedImgs, displacementFields, spectrums, energy
end

function multiresolution(fixedImg, movingImg, displacementSet, quantisations, method,
                         datacost::DataCost, αSet,
                         smooth::SmoothCost, βSet,
                         topology::TopologyCost, χSet,
                         downsample::Integer=2
                        )
    level = length(displacementSet)
    warppedImgs = Vector(level+1)
    displacementFields = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)
    fixedPyramid = gaussian_pyramid(fixedImg, level-1, downsample, 1)

    logger = get_logger(current_module())
    info(logger, "Start multiresolution processing...")
    warppedImgs[1] = copy(movingImg)
    for l = 1:level
        gridDims = size(fixedPyramid[1+level-l])
        warppedPyramid = gaussian_pyramid(warppedImgs[l], level-l, downsample, 1)
        info(logger, "Level $(l): ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridDims)")
        energy[l], spectrums[l] = optimize(fixedPyramid[1+level-l], warppedPyramid[end],
                                           displacementSet[l], quantisations[l],
                                           gridDims, method, datacost, αSet[l], smooth, βSet[l], topology, χSet[l])
        indicator = [indmax(spectrums[l][:,i]) for i in indices(spectrums[l],2)]
        displacementFields[l] = upsample(fieldlize(indicator, displacementSet[l], gridDims), size(fixedImg))
        warppedImgs[l+1] = warp(warppedImgs[l], displacementFields[l])
    end
    info(logger, "Multiresolution processing done!")
    return warppedImgs, displacementFields, spectrums, energy
end
