using NRIRHOPM
using Base.Test

fileDir = dirname(@__FILE__)
include(joinpath(fileDir, "core.jl"))
include(joinpath(fileDir, "potential.jl"))
include(joinpath(fileDir, "cliques.jl"))

# test a simple 5x5 example
fixed = Float64[ 1  2  3  4  5;
                10  9  8  7  6;
                11 12 13 14 15;
                16 17 18 19 20;
                21 22 23 24 25]

moving = Float64[ 1  2  3  4  5;
                 10  9  8 12  6;
                 11  7 13 18 15;
                 16 17 14 19 20;
                 21 22 23 24 25]

deformableWindow = [[i,j] for i in -2:2, j in -2:2]

# with topology preserving
@time x, spectrum = dirhop(fixed, moving, deformableWindow, datacost=SAD(), β=0.1, γ=0.05)

registered, deformgrid = registring(moving, deformableWindow, x)

@test deformgrid[2,4] == [1, -2]
@test deformgrid[3,2] == [-1, 2]
@test deformgrid[3,4] == [1, -1]
@test deformgrid[4,3] == [-1, 1]

@test registered == fixed

# without topology preserving
@time x, spectrum = dirhop(fixed, moving, deformableWindow, datacost=SAD(), β=0.1)

registered, deformgrid = registring(moving, deformableWindow, x)

@test deformed[2,4] == [1, -2]
@test deformed[3,2] == [-1, 2]
@test deformed[3,4] == [1, -1]
@test deformed[4,3] == [-1, 1]

@test registered == fixed

info("All tests passed.")
