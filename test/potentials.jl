import NRIRHOPM: sum_absolute_diff, sum_squared_diff,
                 potts_model, truncated_absolute_diff, truncated_quadratic_diff,
                 topology_preserving, jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ

# construct simple 0-1 images
targetImage = Float64[1 0 1;
                      0 1 0;
                      0 1 1]

sourceImage = Float64[1 0 1;
                      0 1 0;
                      1 1 0]

# create transform vectors
labels = [(i,j) for i in -1:1, j in -1:1]

# test for sum_absolute_diff
info("Testing sum_absolute_diff:")
cost = sum_absolute_diff(targetImage, sourceImage, labels)
@test all(cost .>= 0)

mat = reshape(cost, length(targetImage), length(labels))
dims = size(targetImage)
for 𝒊 in CartesianRange(dims)
    i = sub2ind(dims, 𝒊.I...)
    for a in find(mat[i,:] .== maximum(mat[i,:]))
        𝐭 = CartesianIndex(labels[a])
        @test targetImage[𝒊] == sourceImage[𝒊+𝐭]
    end
end
println("Passed.")

# test for sum_squared_diff
info("Testing sum_squared_diff:")
cost = sum_squared_diff(targetImage, sourceImage, labels)
@test all(cost .>= 0)

mat = reshape(cost, length(targetImage), length(labels))
dims = size(targetImage)
for 𝒊 in CartesianRange(dims)
    i = sub2ind(dims, 𝒊.I...)
    for a in find(mat[i,:] .== maximum(mat[i,:]))
        𝐭 = CartesianIndex(labels[a])
        @test targetImage[𝒊] == sourceImage[𝒊+𝐭]
    end
end
println("Passed.")

# test for potts_model
info("Testing potts_model")
# 2D
fp = (1,2)
fq = fp
d = rand()
@test potts_model(fp, fq, d) == 0
fq = (3,4)
@test potts_model(fp, fq, d) == d

# 3D
fp = (1,2,3)
fq = fp
d = rand()
@test potts_model(fp, fq, d) == 0
fq = (2,3,4)
@test potts_model(fp, fq, d) == d
println("Passed.")


# test for truncated_absolute_diff
info("Testing truncated_absolute_diff:")
# 2D
fp = tuple(rand(2)...)
fq = tuple(rand(2)...)

cost = truncated_absolute_diff(fp, fq, 1, Inf)
delta = abs(vecnorm([fp...] - [fq...]))
@test cost == delta

rate = rand()
@test truncated_absolute_diff(fp, fq, rate, Inf) == rate * delta

# d = 0
@test truncated_absolute_diff(fp, fq, 2, 0.0) == 0

# 3D
fp = tuple(rand(3)...)
fq = tuple(rand(3)...)

cost = truncated_absolute_diff(fp, fq, 1, Inf)
delta = abs(vecnorm([fp...] - [fq...]))
@test cost == delta

rate = rand()
@test truncated_absolute_diff(fp, fq, rate, Inf) == rate * delta

# d = 0
@test truncated_absolute_diff(fp, fq, 2, 0) == 0
println("Passed.")

# test for truncated_quadratic_diff
info("Testing truncated_quadratic_diff:")
# 2D
fp = tuple(rand(2)...)
fq = tuple(rand(2)...)

cost = truncated_quadratic_diff(fp, fq, 1, Inf)
delta = vecnorm([fp...] - [fq...])^2
@test cost - delta < 1e-10

rate = rand()
@test truncated_quadratic_diff(fp, fq, rate, Inf) - rate * delta < 1e-10

# d = 0
@test truncated_quadratic_diff(fp, fq, 2, 0) == 0

# 3D
fp = tuple(rand(3)...)
fq = tuple(rand(3)...)

cost = truncated_quadratic_diff(fp, fq, 1, Inf)
delta = vecnorm([fp...] - [fq...])^2
@test cost - delta < 1e-10

rate = rand()
@test truncated_quadratic_diff(fp, fq, rate, Inf) - rate * delta < 1e-10

# d = 0
@test truncated_quadratic_diff(fp, fq, 2, 0) == 0
println("Passed.")

# test for topology_preserving
info("Testing topology_preserving:")
# topology_preserving                        y
#   □ ⬓ □        ⬓                ⬓          ↑        ⬔ => p1 => a
#   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦        |        ▦ => p2 => b
#   □ ⬓ □              ⬓    ⬓          (x,y):+--> x   ⬓ => p3 => c
#              Jᵇᶠ   Jᵇᵇ    Jᶠᵇ   Jᶠᶠ

# jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ
#   □ ⬓ □        ⬓                ⬓    (y,x):+--> x   ⬔ => p1 => a
#   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦        |        ▦ => p2 => b
#   □ ⬓ □              ⬓    ⬓                ↓        ⬓ => p3 => c
#              Jᵇᵇ   Jᵇᶠ    Jᶠᶠ   Jᶠᵇ        y

# test for Jᵇᶠ
p1 = rand(0:256, 2)
p2 = p1 - [1,0]
p3 = p1 + [0,1]

a, b, c = [1,1], [0,-1], [-1,1]
@test topology_preserving(p2, p1, p3, b, a, c) == 0
@test jᵇᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 0

a, b, c = [-1,-1], [0,-1], [-1,1]
@test topology_preserving(p2, p1, p3, b, a, c) == 1
@test jᵇᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 1

for i = 1:1000
    a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
    @test topology_preserving(p2, p1, p3, b, a, c) == jᵇᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1]))
end

# test for Jᵇᵇ
p1 = rand(0:256, 2)
p2 = p1 - [1,0]
p3 = p1 - [0,1]

a, b, c = [1,-1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 0
@test jᵇᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 0

a, b, c = [-1,-1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 1
@test jᵇᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 1

for i = 1:1000
    a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
    @test topology_preserving(p2, p1, p3, b, a, c) == jᵇᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1]))
end

# test for Jᶠᵇ
p1 = rand(0:256, 2)
p2 = p1 + [1,0]
p3 = p1 - [0,1]

a, b, c = [-1,1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 0
@test jᶠᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 0

a, b, c = [1,-1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 1
@test jᶠᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 1

for i = 1:1000
    a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
    @test topology_preserving(p2, p1, p3, b, a, c) == jᶠᵇ((a[2],a[1]), (b[2],b[1]), (c[2],c[1]))
end

# test for Jᶠᶠ
p1 = rand(0:256, 2)
p2 = p1 + [1,0]
p3 = p1 + [0,1]

a, b, c = [-1,-1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 0
@test jᶠᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 0

a, b, c = [1,1], [0,0], [0,0]
@test topology_preserving(p2, p1, p3, b, a, c) == 1
@test jᶠᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1])) == 1

for i = 1:1000
    a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
    @test topology_preserving(p2, p1, p3, b, a, c) == jᶠᶠ((a[2],a[1]), (b[2],b[1]), (c[2],c[1]))
end

println("Passed.")
