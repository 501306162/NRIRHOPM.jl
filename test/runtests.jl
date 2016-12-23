using NRIRHOPM
using Base.Test

fileDir = dirname(@__FILE__)
include(joinpath(fileDir, "potentials.jl"))
include(joinpath(fileDir, "neighbors.jl"))
include(joinpath(fileDir, "assets.jl"))
include(joinpath(fileDir, "tensors.jl"))
include(joinpath(fileDir, "cliques.jl"))
include(joinpath(fileDir, "hopm.jl"))

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

labels = [(i,j) for i in -2:2, j in -2:2]

# with topology preserving
@time score, v, spectrum = dirhop(fixed, moving, labels, α=0.1, β=0.01)
@show score

registered, deformgrid = registering(moving, labels, v)

@test deformgrid[2,4] == (1, -2)
@test deformgrid[3,2] == (-1, 2)
@test deformgrid[3,4] == (1, -1)
@test deformgrid[4,3] == (-1, 1)

@test registered == fixed

# without topology preserving
@time score, v, spectrum = dirhop(fixed, moving, labels, α=0.1, β=0)
@show score

registered, deformgrid = registering(moving, labels, v)

@test deformgrid[2,4] == (1, -2)
@test deformgrid[3,2] == (-1, 2)
@test deformgrid[3,4] == (1, -1)
@test deformgrid[4,3] == (-1, 1)

@test registered == fixed

info("All tests passed.")
