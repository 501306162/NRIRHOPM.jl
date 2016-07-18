# Test mutual_info.jl
using Base.Test
# load source code
srcpath = realpath(joinpath(dirname(@__FILE__), "../../src/datacost/mutual_info.jl"))
include(srcpath)
