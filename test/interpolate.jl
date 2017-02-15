using Interpolations
import FixedSizeArrays: Vec

@testset "sample" begin
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

@testset "warp" begin
    fixed = [1 1 2 2;
             1 1 2 2;
             3 3 4 4;
             3 3 4 4]
    moving = [4 4 2 2;
              4 4 2 2;
              3 3 1 1;
              3 3 1 1]
    displacementField = [Vec(1,1) Vec(0,0);
                         Vec(0,0) Vec(-1,-1)]
    warpped = warp(moving, displacementField, Constant())
    @test warpped == fixed
end
