using NRIRHOPM, Memento
using Base.Test

set_level(loggerHOPMReg, "warn")

include("funcs.jl")
include("potentials.jl")
include("neighbors.jl")
include("tensors.jl")
include("cliques.jl")
include("hopms.jl")
include("multilevel.jl")
