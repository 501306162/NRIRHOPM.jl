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
    displacements = [SVector(i,j) for i in -1:1, j in -1:1]
    imageDims = indices(targetImage)

    @testset "sum_diff_exp" begin
        FloatDisplacements = [SVector(i,j) for i in -1:0.5:1, j in -1:0.5:1]
        cost = sadexp(targetImage, sourceImage, FloatDisplacements)
        @test all(cost .>= 0)
    end

    @testset "sadexp" begin
        cost = sadexp(targetImage, sourceImage, displacements)
        @test all(cost .>= 0)
        for 𝒊 in CartesianRange(imageDims)
            i = sub2ind(imageDims, 𝒊.I...)
            for a in find(cost[:,i] .== maximum(cost[:,i]))
                𝐝 = SVector(𝒊) + displacements[a]
                @test targetImage[𝒊] == sourceImage[𝐝...]
            end
        end
    end

    @testset "ssdexp" begin
        cost = ssdexp(targetImage, sourceImage, displacements)
        @test all(cost .>= 0)
        for 𝒊 in CartesianRange(imageDims)
            i = sub2ind(imageDims, 𝒊.I...)
            for a in find(cost[:,i] .== maximum(cost[:,i]))
                𝐝 = SVector(𝒊) + displacements[a]
                @test targetImage[𝒊] == sourceImage[𝐝...]
            end
        end
    end

    @testset "potts" begin
        for dim = 1:N
            fp = @SVector rand(dim)
            fq = fp
            d = rand(Float32)
            @test potts(fp, fq, d) == 0
            fq = @SVector rand(dim)
            @test potts(fp, fq, d) == d
            @test pottsexp(fp, fq, d) == e^-d
        end
    end

    @testset "tad" begin
        for dim = 1:N
            fp = @SVector rand(dim)
            fq = @SVector rand(dim)
            rate = rand(Float32)
            @test tad(fp, fq, rate, Inf) ≈ rate * hypot(fp-fq...)
            @test tad(fp, fq, rand(), 0) == 0
            @test tadexp(fp, fq, rand(), 0) == 1
        end
    end

    @testset "tqd" begin
        for dim = 1:N
            fp = @SVector rand(dim)
            fq = @SVector rand(dim)
            rate = rand(Float32)
            @test tqd(fp, fq, rate, Inf) ≈ rate * hypot(fp-fq...)^2
            @test tqd(fp, fq, rand(), 0) == 0
            @test tqdexp(fp, fq, rand(), 0) == 1
        end
    end

    @testset "topology preserving" begin
        # 1 => topology preserving, 0 => otherwise.
        #Refer to the following paper for further details:
        # Cordero-Grande, Lucilio, et al. "A Markov random field approach for
        # topology-preserving registration: Application to object-based tomographic image
        # interpolation." IEEE Transactions on Image Processing 21.4 (2012): 2051.
        function topology_preserving(s₁, s₂, s₃, a, b, c)
            𝐤s₁, 𝐤s₂, 𝐤s₃ = s₁ + a, s₂ + b, s₃ + c
            ∂φ₁∂φ₂ = (𝐤s₂[1] - 𝐤s₁[1]) * (𝐤s₂[2] - 𝐤s₃[2])
            ∂φ₂∂φ₁ = (𝐤s₂[2] - 𝐤s₁[2]) * (𝐤s₂[1] - 𝐤s₃[1])
            ∂r₁∂r₂ = (s₂[1] - s₁[1])*(s₂[2] - s₃[2])
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
        @testset "trivial-2D" begin
            # test for Jᵇᶠ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [-1,-1]; a1 = @SVector [ 1, 1]
            p2 = p1 - [1,0];              b0 = @SVector [ 0,-1]; b1 = @SVector [ 0,-1]
            p3 = p1 + [0,1];              c0 = @SVector [-1, 1]; c1 = @SVector [-1, 1]
            @test jᵇᶠ(a0, b0, c0) == topology_preserving(p2, p1, p3, b0, a0, c0) == 0
            @test jᵇᶠ(a1, b1, c1) == topology_preserving(p2, p1, p3, b1, a1, c1) == 1

            # test for Jᵇᵇ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [-1,-1]; a1 = @SVector [1,-1]
            p2 = p1 - @SVector [1,0];     b0 = @SVector [ 0, 0]; b1 = @SVector [0, 0]
            p3 = p1 - @SVector [0,1];     c0 = @SVector [ 0, 0]; c1 = @SVector [0, 0]
            @test jᵇᵇ(a0, b0, c0) == topology_preserving(p2, p1, p3, b0, a0, c0) == 0
            @test jᵇᵇ(a1, b1, c1) == topology_preserving(p2, p1, p3, b1, a1, c1) == 1

            # test for Jᶠᵇ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [1,-1]; a1 = @SVector [-1, 1]
            p2 = p1 + @SVector [1,0];     b0 = @SVector [0, 0]; b1 = @SVector [ 0, 0]
            p3 = p1 - @SVector [0,1];     c0 = @SVector [0, 0]; c1 = @SVector [ 0, 0]
            @test jᶠᵇ(a0, b0, c0) == topology_preserving(p2, p1, p3, b0, a0, c0) == 0
            @test jᶠᵇ(a1, b1, c1) == topology_preserving(p2, p1, p3, b1, a1, c1) == 1

            # test for Jᶠᶠ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [1, 1]; a1 = @SVector [-1,-1]
            p2 = p1 + @SVector [1,0];     b0 = @SVector [0, 0]; b1 = @SVector [ 0, 0]
            p3 = p1 + @SVector [0,1];     c0 = @SVector [0, 0]; c1 = @SVector [ 0, 0]
            @test jᶠᶠ(a0, b0, c0) == topology_preserving(p2, p1, p3, b0, a0, c0) == 0
            @test jᶠᶠ(a1, b1, c1) == topology_preserving(p2, p1, p3, b1, a1, c1) == 1
        end

        # coordinate system(r,c,z):
        #  up  r     c --->        z × × (front to back)
        #  to  |   left to right     × ×
        # down ↓
        # coordinate => point => label:
        # iii => p1 => α   jjj => p2 => β   kkk => p3 => χ   mmm => p5 => δ
        @testset "trivial-3D" begin
            a = @SVector [0,0,0]
            b = @SVector [0,0,0]
            c = @SVector [0,0,0]
            d = @SVector [0,0,0]
            @test jᶠᶠᶠ(a,b,c,d) == jᵇᶠᶠ(a,b,c,d) == jᶠᵇᶠ(a,b,c,d) == jᵇᵇᶠ(a,b,c,d) == 1
            @test jᶠᶠᵇ(a,b,c,d) == jᵇᶠᵇ(a,b,c,d) == jᶠᵇᵇ(a,b,c,d) == jᵇᵇᵇ(a,b,c,d) == 1
            @test jᶠᶠᶠ(SVector( 1, 1, 1),b,c,d) == 0
            @test jᵇᶠᶠ(SVector(-1, 1, 1),b,c,d) == 0
            @test jᶠᵇᶠ(SVector( 1,-1, 1),b,c,d) == 0
            @test jᵇᵇᶠ(SVector(-1,-1, 1),b,c,d) == 0
            @test jᶠᶠᵇ(SVector( 1, 1,-1),b,c,d) == 0
            @test jᵇᶠᵇ(SVector(-1, 1,-1),b,c,d) == 0
            @test jᶠᵇᵇ(SVector( 1,-1,-1),b,c,d) == 0
            @test jᵇᵇᵇ(SVector(-1,-1,-1),b,c,d) == 0
        end
    end
end
