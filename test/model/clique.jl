@testset "clique" begin
    imageDims = (5,5)
    fixedImg = rand(imageDims)
    movingImg = rand(imageDims)
    displacements = [SVector(i,j) for i in -1:1, j in -1:1]
    labelNum = length(displacements)
    weight = rand()

    @testset "default" begin
        s = @inferred clique(fixedImg, movingImg, displacements, SAD())
        @test size(s) == (length(displacements), prod(imageDims))

        bt = @inferred clique(C8Pairwise(), imageDims, displacements, TAD())
        @test size(bt) == ntuple(x->isodd(x) ? labelNum : prod(imageDims), Val{4})
        @test size(bt.vals) == (labelNum, labelNum)

        cbt = @inferred clique(C8Topology(), imageDims, displacements, TP2D())
        @test size(cbt) == ntuple(x->isodd(x) ? labelNum : prod(imageDims), Val{6})
        @test size(cbt.valBlocks[]) == (labelNum, labelNum, labelNum)
    end

    @testset "custom" begin
        eval(quote
            if !isdefined(:FooRand)
                type FooRand{F<:Function,T<:Real} <: DataCost
                    f::F
                    x::T
                end
                foorand(f, m, d, g, x) = x * rand(length(d), length(f))
                (m::FooRand)(x...) = m.f(x..., m.x)
            end
        end)
        r = clique(fixedImg, movingImg, displacements, FooRand(foorand,10))
        @test size(r) == (length(displacements), prod(imageDims))
    end
end
