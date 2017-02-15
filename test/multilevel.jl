@testset "multilevel" begin
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

    displacements0 = [(i,j) for i in -2:2, j in -2:2]
    displacements1 = [(i,j) for i in -1:1, j in -1:1]
    displacementSet = [displacements0, displacements1]
    gridSet = [(3,3), (5,5)]
    originΔ = vecnorm(moving-fixed)
    # with topology preservation
    warpped, d, spec, energy = multilevel(fixed, moving, displacementSet, gridSet, topology=TP2D(), β=0.1, χ=0.1)
    topologyΔ = vecnorm(warpped[end]-fixed)
    @test topologyΔ < originΔ

    # without topology preservation
    warpped, d, spec, energy = multilevel(fixed, moving, displacementSet, gridSet, topology=TP2D(), β=0.1, χ=0.0)
    smoothΔ = vecnorm(warpped[end]-fixed)
    @test smoothΔ < originΔ
    @show originΔ, smoothΔ, topologyΔ
end
