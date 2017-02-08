module NRIRHOPM

using Combinatorics
using Memento

using Interpolations
import FixedSizeArrays: Vec
export Vec

using NIfTI
using Unitful
using Ranges
using Images


# potential
export AbstractPotential,
       UnaryPotential, DataTerm, DataCost,
       PairwisePotential, SmoothTerm, SmoothCost, RegularTerm,
       TreyPotential, TopologyCost2D,
       QuadraPotential, TopologyCost3D,
       TopologyCost
export SAD, SSD,
       Potts, TAD, TQD,
       TP2D, TP3D

# tensor
export AbstractSymmetricSparseTensor, AbstractTensorBlockBSSTensor
export ValueBlock, IndexBlock, BlockedTensor
export contract, ⊙

# export Connected4, Connected8, Connected6, Connected26, neighbors
# export unaryclique, pairwiseclique, treyclique, quadraclique
# export optimize, warp, upsample, multilevel
export loggerHOPMReg

include("utility.jl")
include("io.jl")
include("tensor.jl")
# include("hopms.jl")
# include("neighbors.jl")
# include("types.jl")
include("potentials.jl")
# include("cliques.jl")
# include("multilevel.jl")

loggerHOPMReg = basic_config("notice"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
