@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  labels::AbstractArray, gridDims::NTuple{N}, method::AbstractHOPMMethod,
                                  data::DataCost, α::Real,
                                  smooth::SmoothCost, β::Real
                                 )
    fixedArgs = [:(getfield(method, $i)) for i = 1:nfields(method)]
    func = shift!(fixedArgs)
    pneighbor = N == 3 ? :(C26Pairwise()) : :(C8Pairwise())
    ret = quote
        logger = get_logger(current_module())
        info(logger, "Creating data cost with weight=$α: ")
        @timelog datacost = clique(fixedImg, movingImg, labels, data, gridDims, α)

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, gridDims, labels, smooth, β)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, rand(eltype(datacost), size(datacost)), $(fixedArgs...))
    end
end

@generated function optimize{T,N}(fixedImg::AbstractArray{T,N}, movingImg::AbstractArray{T,N},
                                  labels::AbstractArray, gridDims::NTuple{N}, method::AbstractHOPMMethod,
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
        @timelog datacost = clique(fixedImg, movingImg, labels, data, α)

        info(logger, "Creating smooth cost with weight=$β: ")
        @timelog smoothcost = clique($pneighbor, gridDims, labels, smooth, β)

        info(logger, "Creating topology cost with weight=$χ: ")
        @timelog topologycost = clique($tneighbor, gridDims, labels, topology, χ)

        info(logger, "Optimizing via High Order Power Method: ")
        @timelog energy, spectrum = $func(datacost, smoothcost, topologycost, rand(eltype(datacost), size(datacost)), $(fixedArgs...))
    end
end
