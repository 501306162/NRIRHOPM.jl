@testset "multiscale" begin
    fixed = [ 1  2  3  4  5;
              6  7  8  9 10;
             11 12 13 14 15;
             16 17 18 19 20;
             21 22 23 24 25]

    moving = [ 1  2  3  4  5;
               6  7  8  9 10;
              11 12 19 14 15;
              16 13 18 17 20;
              21 22 23 24 25]

    displacements0 = [SVector(i,j) for i in -2:2, j in -2:2]
    displacements1 = [SVector(i,j) for i in -1:1, j in -1:1]
    displacementSet = [displacements0, displacements1]
    gridSet = [(2,2), (5,5)]
    @testset "multilevel" begin
        originΔ = vecnorm(moving-fixed)

        # without topology preservation
        warpped, d, spec, energy = multilevel(fixed, moving, displacementSet, gridSet, MixHOPM(), SAD(), [1,1], TAD(), [0.5,0.5])
        smoothΔ = vecnorm(warpped[end]-fixed)
        @test smoothΔ < originΔ

        # # with topology preservation
        warpped, d, spec, energy = multilevel(fixed, moving, displacementSet, gridSet, MixHOPM(), SAD(), [1,1], TAD(), [0.5,0.5], TP2D(), [0.5,0.5])
        topologyΔ = vecnorm(warpped[end]-fixed)
        @test topologyΔ < originΔ

        @show originΔ, smoothΔ, topologyΔ
    end

    @testset "multiresolution" begin
        originΔ = vecnorm(moving-fixed)

        # without topology preservation
        warpped, d, spec, energy = multiresolution(fixed, moving, displacementSet, MixHOPM(), SAD(), [1,1], TAD(), [0.5,0.5])
        smoothΔ = vecnorm(warpped[end]-fixed)
        @test smoothΔ < originΔ

        # with topology preservation
        warpped, d, spec, energy = multiresolution(fixed, moving, displacementSet, MixHOPM(), SAD(), [1,1], TAD(), [0.5,0.5], TP2D(), [0.5,0.5])
        topologyΔ = vecnorm(warpped[end]-fixed)
        @test topologyΔ < originΔ

        @show originΔ, smoothΔ, topologyΔ
    end
end
