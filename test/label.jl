@testset "label" begin
    @testset "fieldlize" begin
        @test fieldlize([1,2,1,3], [(1,2),(3,4),(5,6)], (2,2)) == DVec2D.([(1,2) (1,2); (3,4) (5,6)])
    end

    @testset "fieldmerge" begin
        displacementField = DVec2D.([( 1,1) ( 1,-1);
                                     (-1,1) (-1,-1)])
        @test fieldmerge([displacementField]) == displacementField

        displacementField2 = DVec2D.([(0.5,0.5) (1,0);
                                      (0.0,1.0) (0,0)])
        @test fieldmerge([displacementField, displacementField2]) == DVec2D.([(1,1) ( 1.0, 0.0);
                                                                              (0,1) (-0.5,-0.5)])
    end
end
