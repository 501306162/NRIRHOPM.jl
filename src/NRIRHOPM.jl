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


# potential & model
export AbstractModel
export UnaryModel, DataCost, DataTerm
export PairwiseModel, SmoothCost, SmoothTerm, RegularTerm
export TreyModel
export QuadraModel
export SAD, SSD
export Potts, TAD, TQD
export TP2D, TP3D, TopologyCost

# neighbor
export Connected4, Connected8, Connected6, Connected26, SquareCubic
export C8Pairwise, C26Pairwise, C8Topology, C26Topology
export neighbors

# clique
export clique

# tensor
export AbstractSymmetricSparseTensor, AbstractTensorBlockBSSTensor
export ValueBlock, IndexBlock, BlockedTensor
export contract, ⊙

# hopm & method
export AbstractMethod, AbstractHOPMMethod
export CanonHOPM, MixHOPM

# optimize
export optimize

# label
export DVec2D, DVec3D, DVec, fieldlize

# interpolate
export upsample, warp

# multiscale
export multilevel

# misc.
export @timelog
export loggerHOPMReg

include("util.jl")
include("model/potential.jl")
include("model/neighbor.jl")
include("model/model.jl")
include("model/clique.jl")
include("optimizer/tensor.jl")
include("optimizer/hopm.jl")
include("optimizer/method.jl")
include("optimizer/optimize.jl")
include("label.jl")
include("interpolate.jl")
include("multiscale.jl")
include("io.jl")

loggerHOPMReg = basic_config("info"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
