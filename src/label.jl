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
