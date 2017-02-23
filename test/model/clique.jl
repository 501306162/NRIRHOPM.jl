@testset "clique" begin
    imageDims = (5,5)
    fixedImg = rand(imageDims)
    movingImg = rand(imageDims)
    displacements = [SVector(i,j) for i in -1:1, j in -1:1]
    weight = rand()

    @testset "default" begin
        s = clique(fixedImg, movingImg, displacements, SAD())
        @test size(s) == (length(displacements), prod(imageDims))

        t = clique(C8Pairwise(), imageDims, displacements, TAD())
        @test size(t) == (length(displacements), prod(imageDims), length(displacements), prod(imageDims))
        @test size(t.valBlocks[]) == (length(displacements), length(displacements))
    end

    @testset "custom" begin
        eval(quote
            if !isdefined(:FooRand)
                type FooRand{F<:Function,T<:Real} <: DataCost
                    f::F
                    x::T
                end
                foorand(f, m, d, g, x) = x * rand(length(d), length(f))
            end
        end)
        r = clique(fixedImg, movingImg, displacements, FooRand(foorand,10))
        @test size(r) == (length(displacements), prod(imageDims))
    end
end
