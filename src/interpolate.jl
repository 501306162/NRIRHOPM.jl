fieldlize(indicator, displacements, ImageDims) = reshape([Vec(displacements[i]) for i in indicator], ImageDims)

function interpolate_spectrum{N}(dimsOut::NTuple{N}, dimsIn::NTuple{N}, spectrum::Matrix)
    spectrumVec = reshape([Vec(spectrum[:,i]) for i = 1:prod(dimsIn)], dimsIn)
    knots = ntuple(x->linspace(1, dimsOut[x], dimsIn[x]), Val{N})
    spectrumVecITP = interpolate(knots, spectrumVec, Gridded(Linear()))
    spectrumInterpolated = zeros(size(spectrum,1), prod(dimsOut))
    for 𝒊 in CartesianRange(dimsOut)
        c = sub2ind(dimsOut, 𝒊.I...)
        spectrumInterpolated[:,c] = collect(spectrumVecITP[𝒊])
    end
    return spectrumInterpolated
end

upsample(dimsUp, dims, spectrum) = interpolate_spectrum(dimsUp, dims, spectrum)
downsample(dimsDown, dims, spectrum) = interpolate_spectrum(dimsDown, dims, spectrum)

function warp{N,T<:Real,Dim}(movingImg, displacementField::Array{Vec{N,T},Dim})
    logger = get_logger(current_module())
    imageDims = size(movingImg)
    gridDims = size(displacementField)
    warppedImg = zeros(movingImg)
    if imageDims != gridDims
        knots = ntuple(x->linspace(1, imageDims[x], gridDims[x]), Val{N})
        displacementITP = interpolate(knots, displacementField, Gridded(Linear()))
        movingImgITP = interpolate(movingImg, BSpline(Linear()), OnGrid())
        for 𝒊 in CartesianRange(imageDims)
            𝐝 = Vec(𝒊.I...) + displacementITP[𝒊]
            warppedImg[𝒊] = movingImgITP[𝐝...]
        end
    else
        for 𝒊 in CartesianRange(imageDims)
            𝒅 = 𝒊 + CartesianIndex(displacementField[𝒊]...)
            if checkbounds(Bool, movingImg, 𝒅)
                warppedImg[𝒊] = movingImg[𝒅]
            else
                warn(logger, "𝒅($𝒅) is outbound, skipped.")
            end
        end
    end
    return warppedImg
end
