@testset "cliques" begin
    imageDims = (5,5)
    fixedImg = rand(imageDims)
    movingImg = rand(imageDims)
    labels = [(i,j) for i in -1:1, j in -1:1]
    weight = rand()

    @testset "unaryclique" begin
        @test unaryclique(fixedImg, movingImg, labels) == unaryclique(fixedImg, movingImg, labels, SAD())
        @test unaryclique(fixedImg, movingImg, labels) == unaryclique(fixedImg, movingImg, labels, SAD(), 1)
    end

    @testset "pairwiseclique" begin
        @test pairwiseclique(imageDims, labels) == pairwiseclique(imageDims, labels, TAD())
        @test pairwiseclique(imageDims, labels) == pairwiseclique(imageDims, labels, TAD(), 1)
    end

    @testset "treyclique" begin
        @test treyclique(imageDims, labels) == treyclique(imageDims, labels, TP2D())
        @test treyclique(imageDims, labels) == treyclique(imageDims, labels, TP2D(), 1)
    end

    @testset "quadraclique" begin
        imageDims = (3,3,3)
        labels = [(i,j,k) for i in 0:1, j in 0:1, k in 0:1]
        @test quadraclique(imageDims, labels) == quadraclique(imageDims, labels, TP3D())
        @test quadraclique(imageDims, labels) == quadraclique(imageDims, labels, TP3D(), 1)
    end
end
