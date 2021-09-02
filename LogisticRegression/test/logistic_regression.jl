using DataFrames

@testset "internal functions" begin
    # multi features
    n = 1000
    β = [-1., -1.]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        X₂=vcat(randn(n).-8, randn(n).+16),
        Y=vcat(ones(n), zeros(n))
    )
    lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
    lrm.argv = β

    @test isapprox(LogisticRegression.sigmoid(0), 0.5, atol=1e-10)

    @test isapprox(LogisticRegression.z(lrm, [5, 10]), β' * [5, 10], atol=1e-10)

    @test isapprox(predict(lrm, [5, 10]), LogisticRegression.sigmoid(β' * [5, 10]), atol=1e-10)

    @test isapprox(
        LogisticRegression.log_likelyhood(lrm),
        (
            1 * log(LogisticRegression.sigmoid(β' * [-5, -8])) +
            (1-1) * log(1 - LogisticRegression.sigmoid(β' * [-5, -8]))
        ) + (
            0 * log(LogisticRegression.sigmoid(β' * [10, 16])) +
            (1-0) * log(1 - LogisticRegression.sigmoid(β' * [10, 16]))
        ),
        atol=1e-5
    )

    # single features
    n = 1000
    β = [-1.67]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        Y=vcat(ones(n), zeros(n))
    )
    lrm = LogisticRegressionModel(df, :Y, :X₁)
    lrm.argv = β

    @test isapprox(predict(lrm, 5), LogisticRegression.sigmoid(β' * [5]), atol=1e-10)

    @test isapprox(
        LogisticRegression.log_likelyhood(lrm),
        (
            1 * log(LogisticRegression.sigmoid(β' * [-5])) +
            (1-1) * log(1 - LogisticRegression.sigmoid(β' * [-5]))
        ) + (
            0 * log(LogisticRegression.sigmoid(β' * [10])) +
            (1-0) * log(1 - LogisticRegression.sigmoid(β' * [10]))
        ),
        atol=1e-3
    )
end

@testset "LogisticRegressionModel" begin
    # multi features
    n = 100
    β = [1., 1.]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        X₂=vcat(randn(n).-8, randn(n).+16),
        Y=vcat(ones(n), zeros(n))
    )
    lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
    lrm.argv = β

    fit!(lrm, η=1e-2, atol=-1e-4)
    @test isapprox(LogisticRegression.log_likelyhood(lrm), -1e-4, atol=1e-4)

    # single features
    n = 1000
    β = [-1.67]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        Y=vcat(ones(n), zeros(n))
    )
    lrm = LogisticRegressionModel(df, :Y, :X₁)
    lrm.argv = β

    fit!(lrm, η=1e-2, atol=-1e-3)
    @test isapprox(LogisticRegression.log_likelyhood(lrm), -1e-3, atol=1e-3)
end
