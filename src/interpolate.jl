function upsample{T<:DVec,N}(displacementField::AbstractArray{T,N}, imageDims::NTuple{N,Int})
    gridDims = size(displacementField)
    # scaleFactors = T(imageDims .÷ gridDims) (pending julia-v0.6)
    scaleFactors = T(map(div, imageDims, gridDims))
    knots = ntuple(x->linspace(1, gridDims[x]*scaleFactors[x], gridDims[x]), Val{N})
    itp = interpolate(knots, [scaleFactors.*𝐝 for 𝐝 in displacementField], Gridded(Linear()))
    scaledField = zeros(T, imageDims)
    scaledDims = convert(NTuple{N,Int}, map(*, gridDims, scaleFactors))
    for 𝒊 in CartesianRange(scaledDims)
        scaledField[𝒊] = itp[𝒊]
    end
    return scaledField
end

function warp(movingImg, displacementField)
    # itp = extrapolate(interpolate(movingImg, BSpline(Linear()), OnGrid()), Flat())
    itp = interpolate(movingImg, BSpline(Linear()), OnGrid())
    warppedImg = similar(movingImg)
    outboudscount = 0
    for 𝒊 in CartesianRange(size(movingImg))
        𝐝 = map(+, 𝒊.I, displacementField[𝒊])
        if checkbounds(Bool, movingImg, 𝐝...)
            warppedImg[𝒊] = itp[𝐝...]
        else
            warn("$𝒊 => $𝐝 outbouds!")
            outboudscount += 1
            outboudscount == 30 && break
        end
    end
    return warppedImg
end
