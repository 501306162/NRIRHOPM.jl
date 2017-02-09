"""
    clique(fixedImg, movingImg, displacements, model)
    clique(fixedImg, movingImg, displacements, model, weight)

Returns the **data cost** which should be a `length(displacements)` by
`length(fixedImg)` Float64 Matrix, the so called "spectrum".
"""
@generated function clique{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, displacements::AbstractArray{NTuple{N}}, model, weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    return :(weight*$func(fixedImg, movingImg, displacements, $(fixedArgs...)))
end

"""
    clique(neighbor, imageDims, displacements, model)
    clique(neighbor, imageDims, displacements, model, weight)

Todo: document this
"""
@generated function clique{N,C}(neighbor::Neighborhood{N,C}, imageDims::NTuple{N}, displacements::AbstractArray{NTuple{N}}, model, weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    ret = quote
        𝓭 = reshape(displacements, length(displacements))
        vals = $func(𝓭, $(fixedArgs...))
        idxs = neighbors(neighbor, imageDims)
        BlockedTensor(map(x->ValueBlock(weight*x), collect(vals)), map(x->IndexBlock(x), collect(idxs)), ntuple(x->isodd(x) ? length(displacements) : prod(imageDims), Val{2*$C}))
    end
end
