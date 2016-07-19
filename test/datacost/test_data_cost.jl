# Test data_cost.jl
using Base.Test
using StatsBase
# load source code
srcpath = realpath(joinpath(dirname(@__FILE__), "../../src/datacost/data_cost.jl"))

# construct simple images
targetImage = Float64[1 0 1;
                      0 1 0;
                      0 1 1]
sourceImage = Float64[1 0 1;
                      0 1 0;
                      1 1 0]

# form transform vectors
deformableWindow = [[i,j] for i in -1:1, j in -1:1]
deformers = reshape(deformableWindow, length(deformableWindow))

# call functions
@test unaryclique(targetImage, sourceImage, deformableWindow) == unaryclique(targetImage, sourceImage, deformableWindow; metric=SAD())

@test unaryclique(targetImage, sourceImage, deformableWindow; metric=MI()) == unaryclique(targetImage, sourceImage, deformableWindow, MI())

@test unaryclique(targetImage, sourceImage, deformableWindow; metric=MI(), β=1) == mutual_info(targetImage, sourceImage, deformers, 1)
