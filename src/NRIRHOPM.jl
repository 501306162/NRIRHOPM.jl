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

# neighbor
export Connected4, Connected8, Connected6, Connected26, SquareCubic
export neighbors

# potential model
export AbstractModel
export UnaryModel, DataCost, DataTerm
export PairwiseModel, SmoothCost, SmoothTerm, RegularTerm
export TreyPotential
export QuadraPotential
export TopologyCost, TopologyCost2D, TopologyCost3D
export SAD, SSD
export Potts, TAD, TQD
export TP2D, TP3D

# clique
# export unaryclique, pairwiseclique, treyclique, quadraclique
# export optimize, warp, upsample, multilevel
export loggerHOPMReg

include("tensor.jl")
include("neighbor.jl")
include("potential.jl")
include("model.jl")
# include("clique.jl")
# include("hopms.jl")
include("util.jl")
include("io.jl")
# include("multilevel.jl")

loggerHOPMReg = basic_config("notice"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
