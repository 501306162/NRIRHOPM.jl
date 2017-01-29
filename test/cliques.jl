@testset "cliques" begin
    fixedImg = rand(5,5)
    movingImg = rand(size(fixedImg))
    labels = [(i,j) for i in -1:1, j in -1:1]
    
    @testset "unaryclique" begin
        @test unaryclique(fixedImg, movingImg, labels) == unaryclique(fixedImg, movingImg, labels, SAD())
    end

    @testset "pairwiseclique" begin
        imageDims = size(fixedImg)
        weight = rand()
        @test pairwiseclique(fixedImg, movingImg, labels, weight) == pairwiseclique(fixedImg, movingImg, labels, weight, TAD())
        @test pairwiseclique(imageDims, reshape(labels, length(labels)), TAD()) == pairwiseclique(imageDims, reshape(labels, length(labels)), TAD())
    end

    @testset "treyclique" begin
        imageDims = size(fixedImg)
        weight = rand()
        @test treyclique(fixedImg, movingImg, labels, weight) == treyclique(fixedImg, movingImg, labels, weight, TP())
        @test treyclique(imageDims, reshape(labels, length(labels)), TP()) == treyclique(imageDims, reshape(labels, length(labels)), TP())
    end
end
