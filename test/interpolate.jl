# @testset "warp" begin
#     movingBig = [ 1  1  2  2  3  3  4  4  5  5;
#                   1  1  2  2  3  3  4  4  5  5;
#                   6  6  7  7  8  8  9  9 10 10;
#                   6  6  7  7  8  8  9  9 10 10;
#                  11 11 12 12 19 19 14 14 15 15;
#                  11 11 12 12 19 19 14 14 15 15;
#                  16 16 13 13 18 18 17 17 20 20;
#                  16 16 13 13 18 18 17 17 20 20;
#                  21 21 22 22 23 23 24 24 25 25;
#                  21 21 22 22 23 23 24 24 25 25]
#     warppedBig = warp(movingBig, displacement)
# end

@testset "sample" begin
    # a trivial one
    inputDims = (5,5)
    outputDims = (10,10)
    spectrum = repeat([1:8;], outer=(1,prod(inputDims)))
    @test upsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;], outer=(1,prod(outputDims)))

    outputDims = (3,3)
    @test downsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;], outer=(1,prod(outputDims)))
end
