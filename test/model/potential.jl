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

    @testset "sadexp" begin
        cost = @inferred sadexp(targetImage, sourceImage, displacements)
        @test all(cost .>= 0)
        for 𝒊 in CartesianRange(imageDims)
            i = sub2ind(imageDims, 𝒊.I...)
            for a in find(cost[:,i] .== maximum(cost[:,i]))
                𝐝 = SVector(𝒊) + displacements[a]
                @test targetImage[𝒊] == sourceImage[𝐝...]
            end
        end
        @testset "scale" begin
            targetScale = [1 1 0 0 1 1;
                           1 1 0 0 1 1;
                           0 0 1 1 0 0;
                           0 0 1 1 0 0;
                           0 0 1 1 1 1;
                           0 0 1 1 1 1]

            sourceScale = [1 1 0 0 1 1;
                           1 1 0 0 1 1;
                           0 0 1 1 0 0;
                           0 0 1 1 0 0;
                           1 1 1 1 0 0;
                           1 1 1 1 0 0]
            costScale = sadexp(targetScale, sourceScale, displacements, (3,3))
            @test all(costScale .>= 0)
            for i in eachindex(targetImage)
                @test find(cost[:,i] .== maximum(cost[:,i])) == find(costScale[:,i] .== maximum(costScale[:,i]))
            end
        end
    end

    @testset "ssdexp" begin
        cost = @inferred ssdexp(targetImage, sourceImage, displacements)
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
            @test @inferred(potts(fp, fq, d)) == 0
            fq = @SVector rand(dim)
            @test @inferred(potts(fp, fq, d)) == d
            @test @inferred(pottsexp(fp, fq, d)) == e^-d
        end
    end

    @testset "tad" begin
        for dim = 1:N
            fp = @SVector rand(dim)
            fq = @SVector rand(dim)
            rate = rand(Float32)
            @test @inferred(tad(fp, fq, rate, Inf)) ≈ rate * hypot(fp-fq...)
            @test @inferred(tad(fp, fq, rand(), 0)) == 0
            @test @inferred(tadexp(fp, fq, rand(), 0)) == 1
        end
    end

    @testset "tqd" begin
        for dim = 1:N
            fp = @SVector rand(dim)
            fq = @SVector rand(dim)
            rate = rand(Float32)
            @test @inferred(tqd(fp, fq, rate, Inf)) ≈ rate * hypot(fp-fq...)^2
            @test @inferred(tqd(fp, fq, rand(), 0)) == 0
            @test @inferred(tqdexp(fp, fq, rand(), 0)) == 1
        end
    end

    @testset "topology preserving" begin
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
            @test @inferred(jᵇᶠ(a0, b0, c0)) == topology_preserving(p2, p1, p3, b0, a0, c0) ≤ 0
            @test @inferred(jᵇᶠ(a1, b1, c1)) == topology_preserving(p2, p1, p3, b1, a1, c1) > 0

            # test for Jᵇᵇ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [-1,-1]; a1 = @SVector [1,-1]
            p2 = p1 - @SVector [1,0];     b0 = @SVector [ 0, 0]; b1 = @SVector [0, 0]
            p3 = p1 - @SVector [0,1];     c0 = @SVector [ 0, 0]; c1 = @SVector [0, 0]
            @test @inferred(jᵇᵇ(a0, b0, c0)) == topology_preserving(p2, p1, p3, b0, a0, c0) ≤ 0
            @test @inferred(jᵇᵇ(a1, b1, c1)) == topology_preserving(p2, p1, p3, b1, a1, c1) > 0

            # test for Jᶠᵇ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [1,-1]; a1 = @SVector [-1, 1]
            p2 = p1 + @SVector [1,0];     b0 = @SVector [0, 0]; b1 = @SVector [ 0, 0]
            p3 = p1 - @SVector [0,1];     c0 = @SVector [0, 0]; c1 = @SVector [ 0, 0]
            @test @inferred(jᶠᵇ(a0, b0, c0)) == topology_preserving(p2, p1, p3, b0, a0, c0) ≤ 0
            @test @inferred(jᶠᵇ(a1, b1, c1)) == topology_preserving(p2, p1, p3, b1, a1, c1) > 0

            # test for Jᶠᶠ
            p1 = @SVector rand(UInt8, 2); a0 = @SVector [1, 1]; a1 = @SVector [-1,-1]
            p2 = p1 + @SVector [1,0];     b0 = @SVector [0, 0]; b1 = @SVector [ 0, 0]
            p3 = p1 + @SVector [0,1];     c0 = @SVector [0, 0]; c1 = @SVector [ 0, 0]
            @test @inferred(jᶠᶠ(a0, b0, c0)) == topology_preserving(p2, p1, p3, b0, a0, c0) ≤ 0
            @test @inferred(jᶠᶠ(a1, b1, c1)) == topology_preserving(p2, p1, p3, b1, a1, c1) > 0
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
            @test @inferred(jᶠᶠᶠ(a,b,c,d)) > 0
            @test @inferred(jᵇᶠᶠ(a,b,c,d)) > 0
            @test @inferred(jᶠᵇᶠ(a,b,c,d)) > 0
            @test @inferred(jᵇᵇᶠ(a,b,c,d)) > 0
            @test @inferred(jᶠᶠᵇ(a,b,c,d)) > 0
            @test @inferred(jᵇᶠᵇ(a,b,c,d)) > 0
            @test @inferred(jᶠᵇᵇ(a,b,c,d)) > 0
            @test @inferred(jᵇᵇᵇ(a,b,c,d)) > 0
            @test @inferred(jᶠᶠᶠ(SVector( 1, 1, 1),b,c,d)) ≤ 0
            @test @inferred(jᵇᶠᶠ(SVector(-1, 1, 1),b,c,d)) ≤ 0
            @test @inferred(jᶠᵇᶠ(SVector( 1,-1, 1),b,c,d)) ≤ 0
            @test @inferred(jᵇᵇᶠ(SVector(-1,-1, 1),b,c,d)) ≤ 0
            @test @inferred(jᶠᶠᵇ(SVector( 1, 1,-1),b,c,d)) ≤ 0
            @test @inferred(jᵇᶠᵇ(SVector(-1, 1,-1),b,c,d)) ≤ 0
            @test @inferred(jᶠᵇᵇ(SVector( 1,-1,-1),b,c,d)) ≤ 0
            @test @inferred(jᵇᵇᵇ(SVector(-1,-1,-1),b,c,d)) ≤ 0
        end
    end
end
