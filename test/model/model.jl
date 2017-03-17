@testset "model" begin
    @testset "UnaryModel" begin
        s = SAD()
        @test s.f == sadexp
        s = SSD()
        @test s.f == ssdexp
    end

    @testset "PairwiseModel" begin
        displacements = [[SVector(i,j) for i in 1:2, j in 1:2]...]
        displaceLen = length(displacements)
        p = Potts()
        vals = p.f(displacements, p.d)
        @test p.d == 1
        @test size(vals) == (displaceLen, displaceLen)

        t = TAD()
        vals = t.f(displacements, t.c, t.d)
        @test t.c == 1.0
        @test t.d == Inf
        @test size(vals) == (displaceLen, displaceLen)

        t = TQD()
        vals = t.f(displacements, t.c, t.d)
        @test t.c == 1.0
        @test t.d == Inf
        @test size(vals) == (displaceLen, displaceLen)
    end

    @testset "TopologyModel" begin
        displacements = [[SVector(i,j) for i in 1:2, j in 1:2]...]
        displaceLen = length(displacements)
        t2d = TP2D()
        vals = t2d.f(displacements)
        @test size(vals[]) == (displaceLen, displaceLen, displaceLen)

        displacements = [[SVector(i,j,k) for i in 1:2, j in 1:2, k in 1:2]...]
        displaceLen = length(displacements)
        t3d = TP3D()
        vals = t3d.f(displacements)
        @test size(vals[]) == (displaceLen, displaceLen, displaceLen, displaceLen)
    end
end
