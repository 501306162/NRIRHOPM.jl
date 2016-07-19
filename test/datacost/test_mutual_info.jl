# Test mutual_info.jl
using Base.Test
using StatsBase
# load source code
srcpath = realpath(joinpath(dirname(@__FILE__), "../../src/datacost/mutual_info.jl"))
include(srcpath)

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

# call function
tensor = mutual_info(targetImage, sourceImage, deformers, 1)

# tensor[i] >= 0  ∀ i ∈ tensor
@test all(tensor .>= 0)

# target[i] == source[d(i)] when fully registrating
tensor = reshape(tensor, 9, 9)
for i = 1:9, t in find(tensor[i,:] .== max(tensor[i,:]...))
    ii = ind2sub((3,3), i)
    vv = deformableWindow[t]
    # dᵢ = i + v
    ddᵢ = collect(ii) + collect(vv)
    dᵢ = sub2ind((3,3), ddᵢ...)
    @test targetImage[i] == sourceImage[dᵢ]
end
