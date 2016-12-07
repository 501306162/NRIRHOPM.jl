using NRIRHOPM
using Base.Test

import NRIRHOPM: sum_absolute_diff, truncated_absolute_diff, topology_preserving

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
fp = rand(2)
fq = rand(2)

cost = truncated_absolute_diff(tuple(fp...), tuple(fq...), 1.0, Inf)
delta = abs(vecnorm([fp...] - [fq...]))
@test cost == delta

rate = rand(1)[]
@test truncated_absolute_diff(tuple(fp...), tuple(fq...), rate, Inf) == rate * delta

# d = 0
@test truncated_absolute_diff(tuple(fp...), tuple(fq...), 2.0, 0.0) == 0
println("Passed.")

# test for topology_preserving
info("Testing topology_preserving:")
@test topology_preserving([3,2], [3,3], [2,3], [0,-1], [1,1], [-1,1]) == 0
@test topology_preserving([3,2], [3,3], [2,3], [0,-1], [-1,-1], [-1,1]) == 1

@test topology_preserving([3,4], [3,3], [2,3], [0,0], [1,-1], [0,-1]) == 0
@test topology_preserving([3,4], [3,3], [2,3], [0,0], [-1,1], [0,-1]) == 1

@test topology_preserving([3,4], [3,3], [4,3], [0,1], [-1,-1], [0,-1]) == 0
@test topology_preserving([3,4], [3,3], [4,3], [0,1], [1,1], [0,-1]) == 1

@test topology_preserving([3,2], [3,3], [4,3], [0,0], [-1,1], [0,1]) == 0
@test topology_preserving([3,2], [3,3], [4,3], [0,0], [1,-1], [0,1]) == 1
println("Passed.")
