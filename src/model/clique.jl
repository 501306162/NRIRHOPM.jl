"""
    clique(fixedImg, movingImg, labels, model)
    clique(fixedImg, movingImg, labels, model, gridDims)
    clique(fixedImg, movingImg, labels, model, gridDims, weight)

Returns the **data cost** which should be a `length(labels)` by
`length(fixedImg)` Float64 Matrix, the so called "spectrum".
"""
clique{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
            labels::AbstractArray, model::AbstractModel, gridDims::NTuple{N,Int}=size(fixedImg),
            weight::Real=1) = weight * model(fixedImg, movingImg, labels, gridDims)


"""
    clique(neighbor, imageDims, labels, model)
    clique(neighbor, imageDims, labels, model, weight)

Todo: document this
"""
@generated function clique{N,C}(neighbor::Neighborhood{N,C}, imageDims::NTuple{N},
                                labels::AbstractArray, model::AbstractModel, weight::Real=1)
    Order = 2*C
    if neighbor <: CnTopology
        ret = quote
            valBlocks = model(reshape(labels, length(labels)))
            idxBlocks = neighbors(neighbor, imageDims)
            CompositeBlockedTensor(map(x->ValueBlock(weight*x), valBlocks), map(x->IndexBlock(x), idxBlocks), ntuple(x->isodd(x) ? length(labels) : prod(imageDims), Val{$Order}))
        end
    else
        ret = quote
            vals = model(reshape(labels, length(labels)))
            idxs = neighbors(neighbor, imageDims)
            BlockedTensor(weight*vals, idxs, ntuple(x->isodd(x) ? length(labels) : prod(imageDims), Val{$Order}))
        end
    end
    ret
end
