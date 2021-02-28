@testset "linear_regression.jl internal function" begin
    # Ŷ = 0.1 + 0.3 x₁ + 0.6 X₂ + 0.9 x₃
    argv = [0.1, 0.3, 0.5, 0.7]

    df = DataFrame(X₁=1:10, X₂=11:20, X₃=21:30)
    df.Ŷ = map((x, y, z)->sum(argv .* [1, x, y, z]), df.X₁, df.X₂, df.X₃)

    lrm = LinearRegressionModel(df, :Ŷ, [:X₁, :X₂, :X₃], argv=argv)

    @test ML101.g(lrm, 1) == sum(argv .* [1, 1, 11, 21])
    @test ML101.ĝ(lrm, 1) == sum(argv .* [1, 1, 11, 21])
    @test isapprox(ML101.loss(lrm), 0, atol=1e-20)
end

@testset "linear_regression.jl multipal regression" begin
    # Ŷ = 0.1 + 0.3 x₁ + 0.6 X₂ + 0.9 x₃
    argv = [0.1, 0.3, 0.5, 0.7]

    df = DataFrame(X₁=rand(10), X₂=rand(10), X₃=rand(10))
    df.Ŷ = map((x, y, z)->sum(argv .* [1, x, y, z]), df.X₁, df.X₂, df.X₃)

    lrm = LinearRegressionModel(df, :Ŷ, [:X₁, :X₂, :X₃])
    fit!(lrm)

    @test all(isapprox.(lrm.argv, argv, atol=1e-2))
    @test isapprox(
        predict(lrm, [0.1, 0.2, 0.3]),
        sum(argv .* [1, 0.1, 0.2, 0.3]),
        atol=1e-2
    )
end

@testset "linear_regression.jl linear regression" begin
    # Ŷ = 0.1 + 0.3 x₁
    argv = [0.1, 0.3]
    df = DataFrame(X₁=rand(10))
    df.Ŷ = map(x->sum(argv .* [1, x]), df.X₁)

    lrm = LinearRegressionModel(df, :Ŷ, :X₁)
    fit!(lrm)

    @test all(isapprox.(lrm.argv, argv, atol=1e-2))
    @test isapprox(predict(lrm, 0.1), sum(argv .* [1, 0.1]), atol=1e-2)
end
