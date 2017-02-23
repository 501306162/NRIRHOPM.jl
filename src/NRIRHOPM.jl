module NRIRHOPM

using Reexport
using Combinatorics
using Memento

using Interpolations
@reexport using StaticArrays

using FileIO
using Images
using NIfTI
using Unitful
using Ranges


# tensor
export AbstractSymmetricSparseTensor, AbstractTensorBlockBSSTensor
export ValueBlock, IndexBlock, BlockedTensor
export contract, ⊙

# neighbor
export Connected4, Connected8, Connected6, Connected26, SquareCubic
export C8Pairwise, C26Pairwise, C8Topology, C26Topology
export neighbors

# potential & model
export AbstractModel
export UnaryModel, DataCost, DataTerm
export PairwiseModel, SmoothCost, SmoothTerm, RegularTerm
export TreyModel
export QuadraModel
export SAD, SSD
export Potts, TAD, TQD
export TP2D, TP3D, TopologyCost

# clique
export clique

# hopm & method
export AbstractMethod, AbstractHOPMMethod
export CanonHOPM, MixHOPM

# interpolate
export downsample, warp

# optimize
export optimize

# multilevel
export multilevel

# misc.
export @timelog
export loggerHOPMReg

include("tensor.jl")
include("model/neighbor.jl")
include("model/potential.jl")
include("model/model.jl")
include("model/clique.jl")
include("optimizer/hopm.jl")
include("interpolate.jl")
include("optimizer/method.jl")
include("optimizer/optimize.jl")
include("util.jl")
include("io.jl")
include("multilevel.jl")

loggerHOPMReg = basic_config("info"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
