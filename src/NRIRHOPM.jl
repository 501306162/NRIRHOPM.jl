module NRIRHOPM

using Base.Cartesian

using Reexport
using Combinatorics
using Memento
using ProgressMeter

using Interpolations
@reexport using StaticArrays

@reexport using FileIO
@reexport using Images
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
export C8Pairwise, C26Pairwise, C8Topology, C26Topology, CnTopology
export neighbors

# clique
export clique

# tensor
export AbstractSymmetricSparseTensor, AbstractTensorBlockBSSTensor
export BlockedTensor, ValueBlock, IndexBlock, CompositeBlockedTensor
export contract, ⊙

# hopm & method
export AbstractMethod, AbstractHOPMMethod
export CanonHOPM, MixHOPM

# optimize
export optimize

# label
export DVec2D, DVec3D, DVec, fieldlize, fieldmerge

# interpolate
export upsample, warp

# multiscale
export multilevel, multiresolution

# io
export readDIRLab

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
include("io/constant.jl")
include("io/io.jl")

loggerHOPMReg = Memento.config("info"; fmt="[ {date} | {level} ]: {msg}")
add_handler(loggerHOPMReg, DefaultHandler("HOPMReg.log", DefaultFormatter("[ {date} | {level} ]: {msg}")))

end # module
