"""
    clique(fixedImg, movingImg, labels, model)
    clique(fixedImg, movingImg, labels, model, gridDims)
    clique(fixedImg, movingImg, labels, model, gridDims, weight)

Returns the **data cost** which should be a `length(labels)` by
`length(fixedImg)` Float64 Matrix, the so called "spectrum".
"""
@generated function clique{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N}, labels::AbstractArray, model::AbstractModel, gridDims::NTuple{N,Int}=size(fixedImg), weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    return :(weight*$func(fixedImg, movingImg, labels, gridDims, $(fixedArgs...)))
end

"""
    clique(neighbor, imageDims, labels, model)
    clique(neighbor, imageDims, labels, model, weight)

Todo: document this
"""
@generated function clique{N,C}(neighbor::Neighborhood{N,C}, imageDims::NTuple{N}, labels::AbstractArray, model::AbstractModel, weight::Real=1)
    fixedArgs = [:(getfield(model, $i)) for i = 1:nfields(model)]
    func = shift!(fixedArgs)
    Order = 2*C
    if neighbor <: CnTopology
        ret = quote
            valBlocks = $func(reshape(labels, length(labels)), $(fixedArgs...))
            idxBlocks = neighbors(neighbor, imageDims)
            CompositeBlockedTensor(map(x->ValueBlock(weight*x), valBlocks), map(x->IndexBlock(x), idxBlocks), ntuple(x->isodd(x) ? length(labels) : prod(imageDims), Val{$Order}))
        end
    else
        ret = quote
            vals = $func(reshape(labels, length(labels)), $(fixedArgs...))
            idxs = neighbors(neighbor, imageDims)
            BlockedTensor(weight*vals, idxs, ntuple(x->isodd(x) ? length(labels) : prod(imageDims), Val{$Order}))
        end
    end
    ret
end
