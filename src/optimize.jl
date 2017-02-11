@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  displacements::Array{NTuple{N}}, imageDims::NTuple{N},
                                  data::DataCost, α::Real,
                                  smooth::SmoothCost, β::Real,
                                  method::AbstractHOPMMethod, spectrum::AbstractMatrix
                                 )
    fixedArgs = [:(getfield(method, $i)) for i = 1:nfields(method)]
    func = shift!(fixedArgs)
    pneighbor = N == 3 ? :(C26Pairwise()) : :(C8Pairwise())
    ret = quote
        logger = get_logger(current_module())
        info(logger, "Creating data cost with weight=$α: ")
        @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
        datacost = size(fixedImg) == imageDims ? datacost : downsample(imageDims, size(fixedImg), datacost)

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, imageDims, displacements, smooth, β)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, spectrum, $(fixedArgs...))

        energy, spectrum
    end
end

@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  displacements::Array{NTuple{N}}, imageDims::NTuple{N},
                                  data::DataCost, α::Real,
                                  smooth::SmoothCost, β::Real,
                                  topology::TopologyCost, χ::Real,
                                  method::AbstractHOPMMethod, spectrum::AbstractMatrix
                                 )
    fixedArgs = [:(getfield(method, $i)) for i = 1:nfields(method)]
    func = shift!(fixedArgs)
    pneighbor = N == 3 ? :(C26Pairwise()) : :(C8Pairwise())
    tneighbor = N == 3 ? :(C26Topology()) : :(C8Topology())
    ret = quote
        logger = get_logger(current_module())
        info(logger, "Creating data cost with weight=$α: ")
        @timelog datacost = clique(fixedImg, movingImg, displacements, data, α)
        datacost = size(fixedImg) == imageDims ? datacost : downsample(imageDims, size(fixedImg), datacost)

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, imageDims, displacements, smooth, β)

        info(logger, "Creating topology cost with weight=$χ: ")
        @timelog topologycost = clique($tneighbor, imageDims, displacements, topology, χ)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, topologycost, spectrum, $(fixedArgs...))

        energy, spectrum
    end
end
