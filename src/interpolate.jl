function upsample{T<:DVec,N}(displacementField::AbstractArray{T,N}, imageDims::NTuple{N,Int})
    gridDims = size(displacementField)
    # scaleFactors = T(imageDims .÷ gridDims) (pending julia-v0.6)
    scaleFactors = T(map(div, imageDims, gridDims))
    knots = ntuple(x->linspace(1, gridDims[x]*scaleFactors[x], gridDims[x]), Val{N})
    itp = interpolate(knots, displacementField, Gridded(Linear()))
    scaledField = zeros(T, imageDims)
    scaledDims = convert(NTuple{N,Int}, map(*, gridDims, scaleFactors))
    for 𝒊 in CartesianRange(scaledDims)
        scaledField[𝒊] = round.(itp[𝒊])
    end
    return scaledField
end

function warp{T,N}(movingImg::AbstractArray{T,N}, displacementField)
    warppedImg = similar(movingImg)
    for 𝒊 in CartesianRange(size(movingImg))
        𝐝 = convert(NTuple{N,Int}, map(+, 𝒊.I, displacementField[𝒊]))
        warppedImg[𝒊] = movingImg[𝐝...]
    end
    return warppedImg
end
