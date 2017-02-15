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

downsample(dimsDown, dims, spectrum) = interpolate_spectrum(dimsDown, dims, spectrum)

function warp{N,T<:Real,D<:Interpolations.Degree}(movingImg, displacementField::Array{Vec{N,T},N}, itpType::D=Linear())
    logger = get_logger(current_module())
    imageDims = size(movingImg)
    gridDims = size(displacementField)
    # scale displacementField
    # Todo: factors = imageDims ./ gridDims (pending julia-v0.6)
    factors = map(x->imageDims[x]/gridDims[x], 1:N)
    scaled = [Vec(map(x->factors[x]*𝐝[x],1:N)) for 𝐝 in displacementField]
    knots = ntuple(x->linspace(1, imageDims[x], gridDims[x]), Val{N})
    displacementITP = interpolate(knots, scaled, Gridded(itpType))
    movingImgITP = interpolate(movingImg, BSpline(Linear()), OnGrid())
    warppedImg = zeros(size(movingImg))
    for 𝒊 in CartesianRange(imageDims)
        # Todo: 𝐝 = 𝒊.I .+ collect(displacementITP[𝒊]) (pending julia-v0.6)
        𝐝 = tuple([𝒊[i]+displacementITP[𝒊][i] for i = 1:N]...)
        if Base.checkbounds_indices(Bool, indices(movingImg), 𝐝)
            warppedImg[𝒊] = movingImgITP[𝐝...]
        else
            warn(logger, "𝐝($𝐝) is outbound, skipped.")
        end
    end
    return warppedImg
end
