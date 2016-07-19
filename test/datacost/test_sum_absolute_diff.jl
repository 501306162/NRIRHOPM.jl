# Test sum_absolute_diff.jl
using Base.Test
# load source code
srcpath = realpath(joinpath(dirname(@__FILE__), "../../src/datacost/sum_absolute_diff.jl"))
include(srcpath)

# construct simple images
targetImage = Float64[1 2;
                      3 4]
sourceImage = Float64[1 3;
                      2 4]

# form transform vectors
deformableWindow = [[i,j] for i in -1:1, j in -1:1]
deformers = reshape(deformableWindow, length(deformableWindow))

# call function
tensor = sum_absolute_diff(targetImage, sourceImage, deformers)

# tensor[i] >= 0  ∀ i ∈ tensor
@test all(tensor .>= 0)

# max(tensor) = 1.1(4 - 1)
@test maximum(tensor) == 1.1(maximum(targetImage) - minimum(sourceImage))

# target[i] == source[d(i)] when fully registrating
tensor = reshape(tensor, 4, 9)
for i = 1:4, t in find(tensor[i,:] .== max(tensor[i,:]...))
    ii = ind2sub((2,2), i)
    vv = deformableWindow[t]
    # dᵢ = i + v
    ddᵢ = collect(ii) + collect(vv)
    dᵢ = sub2ind((2,2), ddᵢ...)
    @test targetImage[i] == sourceImage[dᵢ]
end
