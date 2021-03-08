@testset "internal functions" begin
    @test isapprox(ML101.sigmoid(0), 0.5, atol=1e-10)
end
