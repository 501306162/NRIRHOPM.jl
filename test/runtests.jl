using NRIRHOPM, Memento
using Base.Test

set_level(loggerHOPMReg, "warn")

include("model/potential.jl")
include("model/neighbor.jl")
include("model/model.jl")
include("model/clique.jl")
include("optimizer/tensor.jl")
include("optimizer/hopm.jl")
include("optimizer/method.jl")
include("optimizer/optimize.jl")
include("interpolate.jl")
include("multiscale.jl")
include("io/io.jl")
