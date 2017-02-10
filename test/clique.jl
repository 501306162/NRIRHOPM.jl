# define a custom DataCost type for later use
foorand(f, m, d, x) = x * rand(length(d), length(f))
type FooRand{F<:Function,T<:Real} <: DataCost
    f::F
    x::T
end

@testset "clique" begin
    imageDims = (5,5)
    fixedImg = rand(imageDims)
    movingImg = rand(imageDims)
    displacements = [(i,j) for i in -1:1, j in -1:1]
    weight = rand()

    @testset "default" begin
        s = clique(fixedImg, movingImg, displacements, SAD(), weight)
        @test size(s) == (length(displacements), prod(imageDims))

        t = clique(C8Pairwise(), imageDims, displacements, TAD(), weight)
        @test size(t) == (length(displacements), prod(imageDims), length(displacements), prod(imageDims))
        @test size(t.valBlocks[]) == (length(displacements), length(displacements))
    end

    @testset "custom" begin
        r = clique(fixedImg, movingImg, displacements, FooRand(foorand,10), weight)
        @test size(r) == (length(displacements), prod(imageDims))
    end
end
