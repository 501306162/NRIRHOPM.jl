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

# specify δ
δ = 1e2

# call function
tensor = sum_absolute_diff(targetImage, sourceImage, deformers, δ)

# tensor[i] >= 0  ∀ i ∈ tensor
@test all(tensor .>= 0)

# max(tensor) = 4 - 1 + 1e2
@test maximum(tensor) == maximum(targetImage) - minimum(sourceImage) + δ

# target[i] == source[d(i)] when fully registrating
for t in find(tensor .== maximum(tensor))
    tt = ind2sub((4,9), t)
    i = tt[1]
    ii = ind2sub((2,2), tt[1])
    vv = deformableWindow[tt[2]]
    # dᵢ = i + v
    ddᵢ = collect(ii) + collect(vv)
    dᵢ = sub2ind((2,2), ddᵢ...)
    @test targetImage[tt[1]] == sourceImage[dᵢ]
end
