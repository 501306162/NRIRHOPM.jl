using NRIRHOPM, Memento
using Base.Test

set_level(loggerHOPMReg, "warn")

include("tensor.jl")
include("model/neighbor.jl")
include("model/potential.jl")
include("model/model.jl")
include("model/clique.jl")
include("optimizer/hopm.jl")
include("interpolate.jl")
include("optimizer/method.jl")
include("optimizer/optimize.jl")
include("multilevel.jl")
