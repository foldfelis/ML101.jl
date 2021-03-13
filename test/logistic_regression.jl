@testset "internal functions" begin
    n = 1000
    β = [1., 1.]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        X₂=vcat(randn(n).-8, randn(n).+16),
        Y=vcat(zeros(n), ones(n))
    )
    lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
    lrm.argv = β

    @test isapprox(ML101.sigmoid(0), 0.5, atol=1e-10)

    @test isapprox(ML101.z(lrm, [5, 10]), β' * [5, 10], atol=1e-10)

    @test isapprox(predict(lrm, [5, 10]), ML101.sigmoid(β' * [5, 10]), atol=1e-10)

    @test isapprox(
        ML101.log_likelyhood(lrm),
        (
            0 * log(ML101.sigmoid(β' * [-5, -8])) +
            (1-0) * log(1 - ML101.sigmoid(β' * [-5, -8]))
        ) + (
            1 * log(ML101.sigmoid(β' * [10, 16])) +
            (1-1) * log(1 - ML101.sigmoid(β' * [10, 16]))
        ),
        atol=1e-5
    )
end

@testset "LogisticRegressionModel" begin
    n = 100
    β = [0.5, 0.5]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        X₂=vcat(randn(n).-8, randn(n).+16),
        Y=vcat(zeros(n), ones(n))
    )
    lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
    lrm.argv = β

    fit!(lrm, η=1e-2, atol=-1e-4)
    @test isapprox(ML101.log_likelyhood(lrm), -1e-4, atol=1e-4)

    # @df df scatter(:X₁ , :X₂, :Y)
    # x1 = collect(-15:0.1:15)
    # x2 = collect(-15:0.1:15)
    # y = ML101.sigmoid.(lrm.argv[1].*x1 .+ lrm.argv[2].*x2)
    # plot!(x1, x2, y)
end
