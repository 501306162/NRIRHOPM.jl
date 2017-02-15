@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  displacements::Array{NTuple{N}}, gridDims::NTuple{N}, method::AbstractHOPMMethod,
                                  data::DataCost, α::Real,
                                  smooth::SmoothCost, β::Real
                                 )
    fixedArgs = [:(getfield(method, $i)) for i = 1:nfields(method)]
    func = shift!(fixedArgs)
    pneighbor = N == 3 ? :(C26Pairwise()) : :(C8Pairwise())
    ret = quote
        logger = get_logger(current_module())
        info(logger, "Creating data cost with weight=$α: ")
        imageDims = size(fixedImg)
        if imageDims == gridDims
            @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
        else
            # Todo: factors = imageDims ./ gridDims (pending julia-v0.6)
            factors = map(x->imageDims[x]/gridDims[x], 1:N)
            scaled = [tuple(map(x->factors[x]*𝐝[x],1:N)...) for 𝐝 in displacements]
            @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
            datacost = downsample(gridDims, imageDims, datacost)
        end

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, gridDims, displacements, smooth, β)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, rand(eltype(datacost), size(datacost)), $(fixedArgs...))
    end
end

@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  displacements::Array{NTuple{N}}, gridDims::NTuple{N}, method::AbstractHOPMMethod,
                                  data::DataCost, α::Real,
                                  smooth::SmoothCost, β::Real,
                                  topology::TopologyCost, χ::Real,
                                 )
    fixedArgs = [:(getfield(method, $i)) for i = 1:nfields(method)]
    func = shift!(fixedArgs)
    pneighbor = N == 3 ? :(C26Pairwise()) : :(C8Pairwise())
    tneighbor = N == 3 ? :(C26Topology()) : :(C8Topology())
    ret = quote
        logger = get_logger(current_module())
        info(logger, "Creating data cost with weight=$α: ")
        imageDims = size(fixedImg)
        if imageDims == gridDims
            @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
        else
            # Todo: factors = imageDims ./ gridDims (pending julia-v0.6)
            factors = map(x->imageDims[x]/gridDims[x], 1:N)
            scaled = [tuple(map(x->factors[x]*𝐝[x],1:N)...) for 𝐝 in displacements]
            @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
            datacost = downsample(gridDims, imageDims, datacost)
        end

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, gridDims, displacements, smooth, β)

        info(logger, "Creating topology cost with weight=$χ: ")
        @timelog topologycost = clique($tneighbor, gridDims, displacements, topology, χ)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, topologycost, rand(eltype(datacost), size(datacost)), $(fixedArgs...))
    end
end
