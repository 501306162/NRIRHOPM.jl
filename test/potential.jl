import NRIRHOPM: sadexp, ssdexp
import NRIRHOPM: potts, pottsexp, tad, tadexp, tqd, tqdexp
import NRIRHOPM: jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ
import NRIRHOPM: jᶠᶠᶠ, jᵇᶠᶠ, jᶠᵇᶠ, jᵇᵇᶠ, jᶠᶠᵇ, jᵇᶠᵇ, jᶠᵇᵇ, jᵇᵇᵇ

@testset "potential" begin
    N = 3
    targetImage = [1 0 1;
                   0 1 0;
                   0 1 1]

    sourceImage = [1 0 1;
                   0 1 0;
                   1 1 0]
    displacements = [(i,j) for i in -1:1, j in -1:1]
    imageDims = indices(targetImage)

    @testset "sum_diff_exp" begin
        FloatDisplacements = [(i,j) for i in -1:0.5:1, j in -1:0.5:1]
        cost = sadexp(targetImage, sourceImage, FloatDisplacements)
        @test all(cost .>= 0)
    end

    @testset "sadexp" begin
        cost = sadexp(targetImage, sourceImage, displacements)
        @test all(cost .>= 0)
        for 𝒊 in CartesianRange(imageDims)
            i = sub2ind(imageDims, 𝒊.I...)
            for a in find(cost[:,i] .== maximum(cost[:,i]))
                𝐭 = CartesianIndex(displacements[a])
                @test targetImage[𝒊] == sourceImage[𝒊+𝐭]
            end
        end
    end

    @testset "ssdexp" begin
        cost = ssdexp(targetImage, sourceImage, displacements)
        @test all(cost .>= 0)
        for 𝒊 in CartesianRange(imageDims)
            i = sub2ind(imageDims, 𝒊.I...)
            for a in find(cost[:,i] .== maximum(cost[:,i]))
                𝐭 = CartesianIndex(displacements[a])
                @test targetImage[𝒊] == sourceImage[𝒊+𝐭]
            end
        end
    end

    @testset "potts" begin
        for dim = 1:N
            fp = tuple(rand(dim)...)
            fq = fp
            d = rand()
            @test potts(fp, fq, d) == 0
            fq = tuple(rand(dim)...)
            @test potts(fp, fq, d) == d
            @test pottsexp(fp, fq, d) == e^-d
        end
    end

    @testset "tad" begin
        for dim = 1:N
            fpv = rand(dim)
            fqv = rand(dim)
            fp = tuple(fpv...)
            fq = tuple(fqv...)
            rate = rand()
            @test tad(fp, fq, rate, Inf) ≈ rate * hypot(fpv-fqv...)
            @test tad(fp, fq, rand(), 0) == 0
            @test tadexp(fp, fq, rand(), 0) == 1
        end
    end

    @testset "tqd" begin
        for dim = 1:N
            fpv = rand(dim)
            fqv = rand(dim)
            fp = tuple(fpv...)
            fq = tuple(fqv...)
            rate = rand()
            @test tqd(fp, fq, rate, Inf) ≈ rate * hypot(fpv-fqv...)^2
            @test tqd(fp, fq, rand(), 0) == 0
            @test tqdexp(fp, fq, rand(), 0) == 1
        end
    end

    @testset "topology_preserving" begin
        # 1 => topology preserving, 0 => otherwise.
        #Refer to the following paper for further details:
        # Cordero-Grande, Lucilio, et al. "A Markov random field approach for
        # topology-preserving registration: Application to object-based tomographic image
        # interpolation." IEEE Transactions on Image Processing 21.4 (2012): 2051.
        @inline function topology_preserving{T<:Integer}(s₁::Vector{T}, s₂::Vector{T}, s₃::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T})
            @inbounds begin
                𝐤s₁, 𝐤s₂, 𝐤s₃ = s₁ + a, s₂ + b, s₃ + c
                ∂φ₁∂φ₂ = (𝐤s₂[1] - 𝐤s₁[1]) * (𝐤s₂[2] - 𝐤s₃[2])
                ∂φ₂∂φ₁ = (𝐤s₂[2] - 𝐤s₁[2]) * (𝐤s₂[1] - 𝐤s₃[1])
                ∂r₁∂r₂ = (s₂[1] - s₁[1])*(s₂[2] - s₃[2])
            end
            v = (∂φ₁∂φ₂ - ∂φ₂∂φ₁) / ∂r₁∂r₂
            return v > 0 ? 1.0 : 0.0
        end
        # topology_preserving's coordinate system:   y
        #   □ ▦ □        ▦                ▦          ↑        ⬔ => p1 => a
        #   ⬓ ⬔ ⬓  =>  ⬓ ⬔   ⬓ ⬔    ⬔ ⬓   ⬔ ⬓        |        ⬓ => p2 => b
        #   □ ▦ □              ▦    ▦          (x,y):+--> x   ▦ => p3 => c
        #              Jᵇᶠ   Jᵇᵇ    Jᶠᵇ   Jᶠᶠ

        # jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ 's coordinate system:
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

        for i = 1:10
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

        for i = 1:10
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

        for i = 1:10
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

        for i = 1:10
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
