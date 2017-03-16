using MAT
@testset "io" begin
    testimg = open(joinpath(dirname(@__FILE__), "test772.img"), "r") do io
        read(io, Int16, (7,7,2))
    end

    fm = matopen(joinpath(dirname(@__FILE__), "test772.mat"))
    testmat = read(fm, "test")
    @test permutedims(testimg, [2,1,3]) == testmat

    testimgDIR = readDIRLab(joinpath(dirname(@__FILE__), "test772.img"), Dict("Image Dims"=>(7,7,2), "Voxels"=>(1,1,1)))
    @test testimgDIR == testmat

    testnii = load(joinpath(dirname(@__FILE__), "test772.nii"))
    testniiDIR = readDIRLab(testnii)
    @test testniiDIR == testmat
end
