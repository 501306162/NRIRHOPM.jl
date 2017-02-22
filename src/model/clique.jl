"""
    clique(fixedImg, movingImg, labels, model)
    clique(fixedImg, movingImg, labels, model, weight)

Returns the **data cost** which should be a `length(labels)` by
`length(fixedImg)` Float64 Matrix, the so called "spectrum".
"""
@generated function clique{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, labels::AbstractArray, model::AbstractModel, weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    return :(weight*$func(fixedImg, movingImg, labels, $(fixedArgs...)))
end

"""
    clique(neighbor, imageDims, labels, model)
    clique(neighbor, imageDims, labels, model, weight)

Todo: document this
"""
@generated function clique{N,C}(neighbor::Neighborhood{N,C}, imageDims::NTuple{N}, labels::AbstractArray, model::AbstractModel, weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    ret = quote
        𝓭 = reshape(labels, length(labels))
        vals = $func(𝓭, $(fixedArgs...))
        idxs = neighbors(neighbor, imageDims)
        BlockedTensor(map(x->ValueBlock(weight*x), vals), map(x->IndexBlock(x), idxs), ntuple(x->isodd(x) ? length(labels) : prod(imageDims), Val{2*$C}))
    end
end
