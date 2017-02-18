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

downsample(dimsDown, dims, spectrum) = interpolate_spectrum(dimsDown, dims, spectrum)

function warp{N,T<:Real,D<:Interpolations.Degree}(movingImg, displacementField::Array{NTuple{N,T},N}, itpType::D=Linear())
    imageDims = size(movingImg)
    gridDims = size(displacementField)
    # scale displacementField
    # Todo: factors = imageDims ./ gridDims (pending julia-v0.6)
    factors = map(/, imageDims, gridDims)
    scaled = [Vec(map(*, factors, 𝐝)) for 𝐝 in displacementField]
    knots = ntuple(x->linspace(1, imageDims[x], gridDims[x]), Val{N})
    displacementITP = interpolate(knots, scaled, Gridded(itpType))
    movingImgITP = extrapolate(interpolate(movingImg, BSpline(Linear()), OnGrid()), Flat())
    warppedImg = zeros(size(movingImg))
    for 𝒊 in CartesianRange(imageDims)
        𝐝 = Vec(𝒊.I) + displacementITP[𝒊]
        warppedImg[𝒊] = movingImgITP[𝐝...]
    end
    return warppedImg
end
