using NRIRHOPM
using Base.Test

include("potentials.jl")
include("neighbors.jl")
include("assets.jl")
include("tensors.jl")
include("cliques.jl")
include("hopm.jl")

# test a simple 5x5 example
fixed = [ 1  2  3  4  5;
         10  9  8  7  6;
         11 12 13 14 15;
         16 17 18 19 20;
         21 22 23 24 25]

moving = [ 1  2  3  4  5;
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


# test a simple 3x3x3 example
#  1  4  7        111  121  131
#  2  5  8   <=>  211  221  231
#  3  6  9        311  321  331
#-----------front--------------
# 10 13 16        112  122  132
# 11 14 17   <=>  212  222  232
# 12 15 18        312  322  332
#-----------middle-------------
# 19 22 25        113  123  133
# 20 23 26   <=>  213  223  233
# 21 24 27        313  323  333
#-----------back---------------
# fixed = reshape([1:27;], 3, 3, 3)
# moving = copy(fixed)
# moving[1,3,1] = 14
# moving[2,2,2] = 7
#
# labels = [(i,j,k) for i in -1:1, j in -1:1, k in -1:1]
#
# # with topology preserving
# @time score, v, spectrum = dirhop(fixed, moving, labels, α=0.1, β=0.01)
# @show score
#
# registered, deformgrid = registering(moving, labels, v)
#
# @test deformgrid[2,2,2] == (-1, 1, -1)
# @test deformgrid[1,3,1] == (1, -1, 1)
#
# @test registered == fixed


info("All tests passed.")
