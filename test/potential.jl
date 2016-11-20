using NRIRHOPM
using Base.Test

import NRIRHOPM: sum_absolute_diff, truncated_absolute_diff

# construct simple 0-1 images
targetImage = Float64[1 0 1;
                      0 1 0;
                      0 1 1]

sourceImage = Float64[1 0 1;
                      0 1 0;
                      1 1 0]

# create transform vectors
deformableWindow = [[i,j] for i in -1:1, j in -1:1]
deformers = reshape(deformableWindow, length(deformableWindow))

# test for sum_absolute_diff
info("Testing sum_absolute_diff:")
H¹ = sum_absolute_diff(targetImage, sourceImage, deformers)
@test all(H¹ .>= 0)    # tensor[i] >= 0  ∀ i ∈ tensor

mat = reshape(H¹, length(targetImage), length(deformers))
dims = size(targetImage)
for ii in CartesianRange(dims)
    i = sub2ind(dims, ii.I...)
    for a in find(mat[i,:] .== maximum(mat[i,:]))
        dᵢ = collect(ii.I) + deformers[a]
        @test targetImage[i] == sourceImage[dᵢ...]
    end
end
println("Passed.")


# test for truncated_absolute_diff
info("Testing truncated_absolute_diff:")
fp = rand(collect(1:1000), 2)
fq = rand(collect(1:1000), 2)

cost = truncated_absolute_diff(fp, fq, 1.0, Inf)
delta = abs(sqrt(fp[1]^2 + fp[2]^2) - sqrt(fq[1]^2 + fq[2]^2))
@test cost == delta

rate = rand(1)[]
@test truncated_absolute_diff(fp, fq, rate, Inf) == rate * delta

# d = 0
@test truncated_absolute_diff(fp, fq, 2.0, 0.0) == 0
println("Passed.")
