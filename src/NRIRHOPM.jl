module NRIRHOPM

using Interpolations
import FixedSizeArrays: Vec

using NIfTI
using Unitful
using Ranges
using Images

export Vec

# potentials
export AbstractPotential,
       UnaryPotential, DataTerm, DataCost,
       PairwisePotential, SmoothTerm, SmoothCost, RegularTerm,
       TreyPotential, TopologyCost2D,
       QuadraPotential, TopologyCost3D,
       TopologyCost
export SAD, SSD,
       Potts, TAD, TQD,
       TP2D, TP3D

export TensorBlock, BSSTensor, SSTensor, ⊙
export Connected4, Connected8, Connected6, Connected26, neighbors
export unaryclique, pairwiseclique, treyclique, quadraclique
export optimize, warp, upsample, multilevel

include("io.jl")
include("tensors.jl")
include("hopms.jl")
include("neighbors.jl")
include("types.jl")
include("potentials.jl")
include("cliques.jl")
include("multilevel.jl")

end # module
