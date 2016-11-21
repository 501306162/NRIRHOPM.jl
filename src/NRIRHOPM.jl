module NRIRHOPM
using StatsBase

export AbstractPotential, UnaryPotential, DataTerm, DataCost,
       PairwisePotential, SmoothTerm, RegularTerm, TreyPotential
export SAD, Potts, TAD, Quadratic, TP
export unaryclique, pairwiseclique, treyclique
export PSSTensor, ⊙, hopm
export dirhop

# pyramids
export ImagePyramid, PyramidType, ComplexSteerablePyramid, LaplacianPyramid, GaussianPyramid
export subband, toimage, update_subband, update_subband!

include("potential.jl")
include("core.jl")
include("unaryclique.jl")
include("pairwiseclique.jl")
include("treyclique.jl")

function dirhop{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}};
    datacost::DataCost=SAD(),
    smooth::SmoothTerm=TAD(),
    trey::TreyPotential=TP(),
    β::Real=1,
    γ::Real=0,
    χ::Real=1,
    δ::Real=Inf
    )
    imageLen = length(fixedImg)
    deformLen = length(deformableWindow)

    @time 𝐇¹ = unaryclique(fixedImg, movingImg, deformableWindow; algorithm=datacost)
	@time 𝐇² = pairwiseclique(fixedImg, movingImg, deformableWindow; algorithm=smooth, ω=β, χ=χ, δ=δ)
    if γ == 0
        @time score, 𝐯 = hopm(𝐇¹, 𝐇²)
    else
        @time 𝐇³ = treyclique(fixedImg, movingImg, deformableWindow; algorithm=trey, ω=γ)
        @time score, 𝐯 = hopm(𝐇¹, 𝐇², 𝐇³)
    end

    𝐌 = reshape(𝐯, imageLen, deformLen)

    return [findmax(𝐌[i,:])[2] for i in 1:imageLen], 𝐌
end

end # module
