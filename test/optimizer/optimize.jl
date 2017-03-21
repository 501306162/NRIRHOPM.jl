@testset "optimize" begin
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
    displacements = [SVector(i,j) for i in -2:2, j in -2:2]
    imageDims = size(fixed)
    pixelNum, displaceNum = prod(imageDims), length(displacements)

    @testset "data+smooth" begin
        energy, spectrum = @inferred optimize(fixed, moving, displacements, imageDims, MixHOPM(), SAD(), 1, TAD(), 0.05)
        indicator = [indmax(spectrum[:,i]) for i in indices(spectrum,2)]
        displacementField = reshape([displacements[i] for i in indicator], imageDims)
        warppedImg = warp(moving, displacementField)
        @test warppedImg == fixed
    end

    @testset "data+smooth+topology" begin
        energy, spectrum = @inferred optimize(fixed, moving, displacements, imageDims, MixHOPM(), SAD(), 1, TAD(), 0.05, TP2D(), 0.01)
        indicator = [indmax(spectrum[:,i]) for i in indices(spectrum,2)]
        displacementField = reshape([displacements[i] for i in indicator], imageDims)
        warppedImg = warp(moving, displacementField)
        @test warppedImg == fixed
    end
end
