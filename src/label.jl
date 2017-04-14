immutable DisplacementVector2D <: FieldVector{Float64}
    u::Float64
    v::Float64
end
StaticArrays.similar_type(::Type{DisplacementVector2D}, ::Type{Float64}, s::Size{(2,)}) = DisplacementVector2D
StaticArrays.one(::Type{DisplacementVector2D}) = ones(2)

immutable DisplacementVector3D <: FieldVector{Float64}
    u::Float64
    v::Float64
    w::Float64
end
StaticArrays.similar_type(::Type{DisplacementVector3D}, ::Type{Float64}, s::Size{(3,)}) = DisplacementVector3D
StaticArrays.one(::Type{DisplacementVector3D}) = ones(3)

typealias DVec2D DisplacementVector2D
typealias DVec3D DisplacementVector3D
typealias DVec Union{DVec2D, DVec3D}

@generated function fieldlize{N}(indicator, labels, dims::NTuple{N})
    f = N == 2 ? DVec2D : DVec3D
    return :(reshape([$f(labels[i]) for i in indicator], dims))
end

function fieldmerge{T<:DVec,N}(displacementSet::Vector{Array{T,N}})
    imageDims = size(displacementSet[])
    meshgrid = Array{Vector}(imageDims)
    for i in CartesianRange(imageDims)
        meshgrid[i] = collect(i.I)
    end
    temp = copy(meshgrid)
    for level = 1:length(displacementSet)
        # itp = interpolate(displacementSet[level], BSpline(Constant()), OnGrid())
        # for i in eachindex(temp)
        #     temp[i] = temp[i] + itp[temp[i]...]
        # end
        d = displacementSet[level]
        for i in eachindex(temp)
            t = Int.(temp[i])
            temp[i] += d[t...]
        end
    end
    T.(temp .- meshgrid)
end
