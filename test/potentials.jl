import NRIRHOPM: sum_absolute_diff, sum_squared_diff,
                 potts_model, truncated_absolute_diff, truncated_quadratic_diff,
                 topology_preserving, jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ, jᶠᶠᶠ, jᵇᶠᶠ, jᶠᵇᶠ, jᵇᵇᶠ,
                 jᶠᶠᵇ, jᵇᶠᵇ, jᶠᵇᵇ, jᵇᵇᵇ

targetImage = Float64[1 0 1;
                      0 1 0;
                      0 1 1]

sourceImage = Float64[1 0 1;
                      0 1 0;
                      1 1 0]

labels = [(i,j) for i in -1:1, j in -1:1]

@testset "potentials" begin
    @testset "sum_absolute_diff" begin
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
    end

    @testset "sum_squared_diff" begin
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
    end

    @testset "potts_model" begin
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
    end

    @testset "truncated_absolute_diff" begin
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
    end

    @testset "truncated_quadratic_diff" begin
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
    end

    @testset "topology_preserving" begin
        # topology_preserving                        y
        #   □ ▦ □        ▦                ▦          ↑        ⬔ => p1 => a
        #   ⬓ ⬔ ⬓  =>  ⬓ ⬔   ⬓ ⬔    ⬔ ⬓   ⬔ ⬓        |        ⬓ => p2 => b
        #   □ ▦ □              ▦    ▦          (x,y):+--> x   ▦ => p3 => c
        #              Jᵇᶠ   Jᵇᵇ    Jᶠᵇ   Jᶠᶠ

        # jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ
        #   □ ⬓ □        ⬓                ⬓      r,c-->    ⬔ => p1 => a
        #   ▦ ⬔ ▦  =>  ▦ ⬔   ▦ ⬔    ⬔ ▦   ⬔ ▦    |         ⬓ => p2 => b
        #   □ ⬓ □              ⬓    ⬓            ↓         ▦ => p3 => c
        #              Jᵇᵇ   Jᶠᵇ    Jᶠᶠ   Jᵇᶠ

        # test for Jᵇᶠ
        p1 = rand(0:256, 2)
        p2 = p1 - [1,0]
        p3 = p1 + [0,1]

        a, b, c = [1,1], [0,-1], [-1,1]
        @test topology_preserving(p2, p1, p3, b, a, c) == 1
        @test jᵇᶠ(tuple(a...), tuple(b...), tuple(c...)) == 1

        a, b, c = [-1,-1], [0,-1], [-1,1]
        @test topology_preserving(p2, p1, p3, b, a, c) == 0
        @test jᵇᶠ(tuple(a...), tuple(b...), tuple(c...)) == 0

        for i = 1:1000
            a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
            @test topology_preserving(p2, p1, p3, b, a, c) == jᵇᶠ(tuple(a...), tuple(b...), tuple(c...))
        end

        # test for Jᵇᵇ
        p1 = rand(0:256, 2)
        p2 = p1 - [1,0]
        p3 = p1 - [0,1]

        a, b, c = [1,-1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 1
        @test jᵇᵇ(tuple(a...), tuple(b...), tuple(c...)) == 1

        a, b, c = [-1,-1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 0
        @test jᵇᵇ(tuple(a...), tuple(b...), tuple(c...)) == 0

        for i = 1:1000
            a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
            @test topology_preserving(p2, p1, p3, b, a, c) == jᵇᵇ(tuple(a...), tuple(b...), tuple(c...))
        end

        # test for Jᶠᵇ
        p1 = rand(0:256, 2)
        p2 = p1 + [1,0]
        p3 = p1 - [0,1]

        a, b, c = [-1,1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 1
        @test jᶠᵇ(tuple(a...), tuple(b...), tuple(c...)) == 1

        a, b, c = [1,-1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 0
        @test jᶠᵇ(tuple(a...), tuple(b...), tuple(c...)) == 0

        for i = 1:1000
            a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
            @test topology_preserving(p2, p1, p3, b, a, c) == jᶠᵇ(tuple(a...), tuple(b...), tuple(c...))
        end

        # test for Jᶠᶠ
        p1 = rand(0:256, 2)
        p2 = p1 + [1,0]
        p3 = p1 + [0,1]

        a, b, c = [-1,-1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 1
        @test jᶠᶠ(tuple(a...), tuple(b...), tuple(c...)) == 1

        a, b, c = [1,1], [0,0], [0,0]
        @test topology_preserving(p2, p1, p3, b, a, c) == 0
        @test jᶠᶠ(tuple(a...), tuple(b...), tuple(c...)) == 0

        for i = 1:1000
            a, b, c = rand(-15:15, 2), rand(-15:15, 2), rand(-15:15, 2)
            @test topology_preserving(p2, p1, p3, b, a, c) == jᶠᶠ(tuple(a...), tuple(b...), tuple(c...))
        end

        # topology preserving in 3D(just some trivial tests)
        # coordinate system(r,c,z):
        #  up  r     c --->        z × × (front to back)
        #  to  |   left to right     × ×
        # down ↓
        # coordinate => point => label:
        # iii => p1 => α   jjj => p2 => β   kkk => p3 => χ   mmm => p5 => δ

        # test for Jᶠᶠᶠ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᶠᶠ(a,b,c,d) == 1

        a, b, c, d = (1,1,1), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᶠᶠ(a,b,c,d) == 0

        # test for Jᵇᶠᶠ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᶠᶠ(a,b,c,d) == 1

        a, b, c, d = (-1,1,1), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᶠᶠ(a,b,c,d) == 0

        # test for Jᶠᵇᶠ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᵇᶠ(a,b,c,d) == 1

        a, b, c, d = (1,-1,1), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᵇᶠ(a,b,c,d) == 0

        # test for Jᵇᵇᶠ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᵇᶠ(a,b,c,d) == 1

        a, b, c, d = (-1,-1,1), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᵇᶠ(a,b,c,d) == 0

        # test for Jᶠᶠᵇ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᶠᵇ(a,b,c,d) == 1

        a, b, c, d = (1,1,-1), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᶠᵇ(a,b,c,d) == 0

        # test for Jᵇᶠᵇ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᶠᵇ(a,b,c,d) == 1

        a, b, c, d = (-1,1,-1), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᶠᵇ(a,b,c,d) == 0

        # test for Jᶠᵇᵇ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᵇᵇ(a,b,c,d) == 1

        a, b, c, d = (1,-1,-1), (0,0,0), (0,0,0), (0,0,0)
        @test jᶠᵇᵇ(a,b,c,d) == 0

        # test for Jᵇᵇᵇ
        a, b, c, d = (0,0,0), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᵇᵇ(a,b,c,d) == 1

        a, b, c, d = (-1,-1,-1), (0,0,0), (0,0,0), (0,0,0)
        @test jᵇᵇᵇ(a,b,c,d) == 0
    end
end
