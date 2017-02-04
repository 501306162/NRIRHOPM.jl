@testset "multilevel" begin
    @testset "optimize-2D" begin
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
        labels = [(i,j) for i in -2:2, j in -2:2]
        dims = size(fixed)

        @testset "without topology preservation" begin
            energy, spectrum = optimize(fixed, moving, dims, labels, SAD(), TAD(), 0.07)
            indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
            displacement = reshape([Vec(labels[i]) for i in indicator], size(fixed))
            warppedImg = warp(moving, displacement)
            @test warppedImg == fixed
            @testset "warp" begin
                movingBig = [ 1  1  2  2  3  3  4  4  5  5;
                              1  1  2  2  3  3  4  4  5  5;
                              6  6  7  7  8  8  9  9 10 10;
                              6  6  7  7  8  8  9  9 10 10;
                             11 11 12 12 19 19 14 14 15 15;
                             11 11 12 12 19 19 14 14 15 15;
                             16 16 13 13 18 18 17 17 20 20;
                             16 16 13 13 18 18 17 17 20 20;
                             21 21 22 22 23 23 24 24 25 25;
                             21 21 22 22 23 23 24 24 25 25]
                warppedBig = warp(movingBig, displacement)
            end
        end
        @testset "with topology preservation" begin
            energy, spectrum = optimize(fixed, moving, dims, labels, SAD(), TAD(), TP2D(), 0.07, 0.01)
            indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
            displacement = reshape([Vec(labels[i]) for i in indicator], size(fixed))
            warppedImg = warp(moving, displacement)
            @test warppedImg == fixed
        end
    end

    @testset "optimize-3D" begin
        #  1  4  7        111  121  131
        #  2  5  8   <=>  211  221  231
        #  3  6  9        311  321  331
        #-----------front--------------
        # 10 13 16        112  122  132
        # 11 14 17   <=>  212  222  232
        # 12 15 18        312  322  332
        #-----------middle-------------
        # 19 22 25        113  123  133
        # 20 23 26   <=>  213  223  233
        # 21 24 27        313  323  333
        #-----------back---------------
        fixed = reshape([1:27;], 3, 3, 3)
        moving = copy(fixed)

        moving[1,3,2] = 14
        moving[2,2,2] = 23
        moving[2,2,3] = 25
        moving[1,3,3] = 16

        labels = [(i,j,k) for i in -1:1, j in -1:1, k in -1:1]
        dims = size(fixed)

        @testset "without topology preservation" begin
            energy, spectrum = optimize(fixed, moving, dims, labels, SAD(), TAD(), 0.07)
            indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
            displacement = reshape([Vec(labels[i]) for i in indicator], size(fixed))
            warppedImg = warp(moving, displacement)
            @test warppedImg == fixed
        end

        if !haskey(ENV, "TRAVIS")    # skip this test on Travis CI because it's very time-consuming
            @testset "with topology preservation" begin
                energy, spectrum = optimize(fixed, moving, dims, labels, SAD(), TAD(), TP3D(), 0.07, 0.01)
                indicator = [indmax(spectrum[i,:]) for i in indices(spectrum,1)]
                displacement = reshape([Vec(labels[i]) for i in indicator], size(fixed))
                warppedImg = warp(moving, displacement)
                @test warppedImg == fixed
            end
        end
    end

    @testset "upsample" begin
        # a trivial one
        inputDims = (5,5)
        outputDims = (10,10)
        spectrum = repeat([1:8;]', outer=(prod(inputDims),1))
        @test upsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;]', outer=(prod(outputDims),1))
        # downsampling also works
        outputDims = (3,3)
        @test upsample(outputDims, inputDims, spectrum) ≈ repeat([1:8;]', outer=(prod(outputDims),1))
    end

    if !haskey(ENV, "TRAVIS")    # skip this test on Travis CI because it's very time-consuming
        fixed = reshape(Float64[1:125;], 5, 5, 5)
        moving = copy(fixed)
        labels = [(i,j,k) for i in -1:1, j in -1:1, k in -1:1]
        movingImgs, displacements, spectrums = multilevel(fixed, moving, [labels, labels], [(3,3,3),(5,5,5)], α=0.01, β=0.0, verbose=true)
        @test movingImgs[end] == fixed
    end
end
