@testset "interpolate" begin
    @testset "warp" begin
        fixed = [1 2;
                 3 4]
        moving = [4 2;
                  3 1]
        displacementField = [(1,1) ( 0, 0);
                             (0,0) (-1,-1)]
        warpped = warp(moving, displacementField)
        @test warpped == fixed
    end
    @testset "upsample" begin
        displacementField = [(1,1) ( 0, 0);
                             (0,0) (-1,-1)]
        @test upsample(DVec2D.(displacementField), (2,2)) == DVec2D.(displacementField)
    end
end
