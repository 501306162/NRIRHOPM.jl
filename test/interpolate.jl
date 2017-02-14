using Interpolations
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
    @testset "trivial" begin
        inputDims = (5,5)
        outputDims = (10,10)
        spectrum = repeat([1:8;], outer=(1,prod(inputDims)))
        @test upsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;], outer=(1,prod(outputDims)))

        outputDims = (3,3)
        @test downsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;], outer=(1,prod(outputDims)))
    end

    @testset "forward-resample" begin
        fixed = [1 1 1 1 2 2 2 2;
                 1 1 1 1 2 2 2 2;
                 1 1 1 1 2 2 2 2;
                 1 1 1 1 2 2 2 2;
                 3 3 3 3 4 4 4 4;
                 3 3 3 3 4 4 4 4;
                 3 3 3 3 4 4 4 4;
                 3 3 3 3 4 4 4 4]

        moving = [4 4 4 4 2 2 2 2;
                  4 4 4 4 2 2 2 2;
                  4 4 4 4 2 2 2 2;
                  4 4 4 4 2 2 2 2;
                  3 3 3 3 1 1 1 1;
                  3 3 3 3 1 1 1 1;
                  3 3 3 3 1 1 1 1;
                  3 3 3 3 1 1 1 1]

        spectrum = sadexp(fixed, moving, [(4,4),(-4,-4)])

        # (8,8) => (4,4)
        spectrumDown = downsample((4,4), (8,8), spectrum)
        @test spectrumDown == [1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0;
                               0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 1]

        # (8,8) => (2,2)
        spectrumDown = downsample((2,2), (8,8), spectrum)
        @test spectrumDown == [1 0 0 0;
                               0 0 0 1]

        # (8,8) => (3,3)
        spectrumDown = downsample((3,3), (8,8), spectrum)
        @test spectrumDown ==  [1 0.5 0 0.5 0.25 0   0  0   0;
                                0 0   0 0   0.25 0.5 0  0.5 1]
    end

    @testset "backward-resample" begin
        fixed = [1 2 3;
                 2 3 4;
                 3 4 5]

        moving = [5 2 3;
                  2 3 4;
                  3 4 1]

        spectrum = sadexp(fixed, moving, [(2,2),(-2,-2)])
        indicatorExpected = [indmax(spectrum[:,i]) for i in indices(spectrum,2)]

        inputDims, outDims = (3,3), (8,8)
        knots = ntuple(x->linspace(1, outDims[x], inputDims[x]), 2)
        fixedITP = interpolate(knots, fixed, Gridded(Linear()))
        movingITP = interpolate(knots, moving, Gridded(Linear()))
        fixedUp = fixedITP[1:8,1:8]
        movingUp = movingITP[1:8,1:8]

        spectrum = sadexp(fixedUp, movingUp, [(2*8/3,2*8/3),(-2*8/3,-2*8/3)])
        spectrumDown = downsample((3,3), (8,8), spectrum)
        indicator = [indmax(spectrumDown[:,i]) for i in indices(spectrumDown,2)]

        @test indicator == indicatorExpected
    end
end
