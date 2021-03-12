@testset "internal functions" begin
    @test isapprox(ML101.sigmoid(0), 0.5, atol=1e-10)

    lrm = LogisticRegressionModel(
        vcat(randn(10).+3, randn(10).+30),
        hcat(
            vcat(randn(10).+1, randn(10).+10),
            vcat(randn(10).+2, randn(10).+20)
        ),
        [3.0, 5.0],
        20
    )
    @test isapprox(ML101.z(lrm, [5, 10]), 3.0*5+5.0*10, atol=1e-10)
end

@testset "LogisticRegressionModel" begin
    df = DataFrame(
        X₁=vcat(randn(10).+1, randn(10).+10),
        X₂=vcat(randn(10).+2, randn(10).+20),
        Y=vcat(randn(10).+3, randn(10).+30)
    )
    lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
    lrm.argv = [3.0, 5.0]

    @test isapprox(predict(lrm, [5, 10]), ML101.sigmoid(3.0*5+5.0*10), atol=1e-10)

    @test isapprox(
        ML101.log_likelyhood(lrm),
        (
            3 * log(ML101.sigmoid([3.0, 5.0]' * [1, 2])) +
            (1-3) * log(1 - ML101.sigmoid([3.0, 5.0]' * [1, 2]))
        ) + (
            30 * log(ML101.sigmoid([3.0, 5.0]' * [10, 20])) +
            (1-30) * log(1 - ML101.sigmoid([3.0, 5.0]' * [10, 20]))
        ),
        atol=1e-10
    )
end
