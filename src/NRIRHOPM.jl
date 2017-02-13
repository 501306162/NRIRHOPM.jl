module NRIRHOPM

using Combinatorics
using Memento

using Interpolations
import FixedSizeArrays: Vec

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
export fieldlize, upsample, downsample, warp

# optimize
export optimize

# multilevel


# misc.
export @timelog
export loggerHOPMReg

include("tensor.jl")
include("neighbor.jl")
include("potential.jl")
include("model.jl")
include("clique.jl")
include("hopm.jl")
include("interpolate.jl")
include("method.jl")
include("optimize.jl")
include("util.jl")
include("io.jl")
# include("multilevel.jl")

loggerHOPMReg = basic_config("notice"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
