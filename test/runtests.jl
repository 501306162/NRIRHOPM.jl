using NRIRHOPM, Memento
using Base.Test

set_level(loggerHOPMReg, "warn")

include("tensor.jl")
include("potential.jl")
include("neighbor.jl")
# include("cliques.jl")
# include("hopms.jl")
# include("multilevel.jl")
