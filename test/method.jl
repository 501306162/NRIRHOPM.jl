@testset "method" begin
    c = CanonHOPM()
    m = MixHOPM()
    @test c.f == hopm_canonical
    @test m.f == hopm_mixed
    @test c.tolerance == m.tolerance == 1e-5
    @test c.maxIteration == m.maxIteration == 300
end
