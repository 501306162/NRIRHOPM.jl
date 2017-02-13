@testset "multilevel" begin
    if !haskey(ENV, "TRAVIS")    # skip this test on Travis CI because it's very time-consuming
        fixed = reshape(Float64[1:125;], 5, 5, 5)
        moving = copy(fixed)
        labels = [(i,j,k) for i in -1:1, j in -1:1, k in -1:1]
        movingImgs, displacements, spectrums = multilevel(fixed, moving, [labels, labels], [(3,3,3),(5,5,5)], β=0.01, χ=0.0)
        @test movingImgs[end] == fixed
    end
end
