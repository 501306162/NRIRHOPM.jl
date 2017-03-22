function multilevel(fixedImg, movingImg, displacementSet, gridSet;
                    method::AbstractHOPMMethod=MixHOPM(),
                    datacost::DataCost=SAD(), αs=ones(length(displacementSet)),
                    smooth::SmoothCost=TAD(), βs=ones(length(displacementSet)),
                    topology::TopologyCost=TP3D(), χ::Real=1
                   )
    logger = get_logger(current_module())
    info(logger, "Start multilevel processing...")
    imageDims = size(fixedImg)
    level = length(displacementSet)
    warppedImgs = Vector(level)
    displacementFields = Vector(level)
    spectrums = Vector(level)
    energy = Vector(level)

    # enforce eltype consistency
    fixedImg = convert(AbstractArray{Float64}, fixedImg)
    movingImg = convert(AbstractArray{Float64}, movingImg)

    if χ != 0
        info(logger, "Level 0: ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridSet[1])")
        energy[1], spectrums[1] = optimize(fixedImg, movingImg, displacementSet[1], gridSet[1], method, datacost, αs[1], smooth, βs[1], topology, χ)
        indicator = [indmax(spectrums[1][:,i]) for i in indices(spectrums[1],2)]
        displacementFields[1] = upsample(fieldlize(indicator, displacementSet[1], gridSet[1]), imageDims)
        warppedImgs[1] = warp(movingImg, displacementFields[1])
    else
        warppedImgs[1] = copy(movingImg)
    end

    for l = 2:level
        info(logger, "Level $(l-1): ")
        info(logger, "Image Dimension: $(size(fixedImg))")
        info(logger, "Grid Dimension: $(gridSet[l])")
        energy[l], spectrums[l] = optimize(fixedImg, warppedImgs[l-1], displacementSet[l], gridSet[l], method, datacost, αs[l], smooth, βs[l])
        indicator = [indmax(spectrums[l][:,i]) for i in indices(spectrums[l],2)]
        displacementFields[l] = upsample(fieldlize(indicator, displacementSet[l], gridSet[l]), imageDims)
        warppedImgs[l] = warp(warppedImgs[l-1], displacementFields[l])
    end

    info(logger, "Multilevel processing done!")
    return warppedImgs, displacementFields, spectrums, energy
end

# function multiresolution(fixedImg, movingImg, displacementSet, gridSet;
#                          method::AbstractHOPMMethod=MixHOPM(),
#                          datacost::DataCost=SAD(), α::Real=1,
#                          smooth::SmoothCost=TAD(), β::Real=1,
#                          topology::TopologyCost=TP3D(), χ::Real=1
#                         )
#
# end
