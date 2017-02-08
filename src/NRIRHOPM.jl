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

# tensor
export AbstractSymmetricSparseTensor, AbstractTensorBlockBSSTensor
export ValueBlock, IndexBlock, BlockedTensor
export contract, ⊙

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

# neighbor
export Connected4, Connected8, Connected6, Connected26, SquareCubic
export neighbors


# export unaryclique, pairwiseclique, treyclique, quadraclique
# export optimize, warp, upsample, multilevel
export loggerHOPMReg

include("tensor.jl")
include("potential.jl")
include("neighbor.jl")
# include("types.jl")
# include("clique.jl")
# include("hopms.jl")
include("util.jl")
include("io.jl")
# include("multilevel.jl")

loggerHOPMReg = basic_config("notice"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
