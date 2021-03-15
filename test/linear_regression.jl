@testset "internal function" begin
    # Ŷ = 0.1 + 0.3 x₁ + 0.6 X₂ + 0.9 x₃
    x₁ = collect(1:10)
    x₂ = collect(11:20)
    x₃ = collect(21:30)
    ŷ = 0.1 .+ 0.3x₁ .+ 0.5x₂ .+ 0.7x₃
    df = DataFrame(Ŷ=ŷ, X₁=x₁, X₂=x₂, X₃=x₃)

    lrm = LinearRegressionModel(df, :Ŷ, [:X₁, :X₂, :X₃])
    lrm.argv = [0.1, 0.3, 0.5, 0.7]

    X = Matrix(collect(df[1, [:X₁, :X₂, :X₃]])')
    @test ML101.y(lrm, X)[1] == 0.1 + 0.3*1 + 0.5*11 + 0.7*21
    @test ML101.loss(lrm) == 0
end

@testset "multiple regression" begin
    # Ŷ = 0.1 + 0.3 x₁ + 0.5 X₂ + 0.7 x₃
    x₁ = rand(10)
    x₂ = rand(10)
    x₃ = rand(10)
    ϵ = randn(10) * 1e-20
    y = 0.1 .+ 0.3x₁ .+ 0.5x₂ .+ 0.7x₃ .+ ϵ
    df = DataFrame(Y=y, X₁=x₁, X₂=x₂, X₃=x₃)

    lrm = LinearRegressionModel(df, :Y, [:X₁, :X₂, :X₃])
    fit!(lrm, atol=1e-7)

    @test all(isapprox.(lrm.argv, [0.1, 0.3, 0.5, 0.7], atol=1e-2))
    @test isapprox(
        predict(lrm, [0.1, 0.2, 0.3]),
        0.1*1 + 0.3*0.1 + 0.5*0.2 + 0.7*0.3,
        atol=1e-2
    )
end

@testset "simple linear regression" begin
    # Ŷ = 0.1 + 0.3 x₁
    x = rand(10)
    ϵ = randn(10) * 1e-20
    ŷ = 0.1 .+ 0.3x + ϵ
    df = DataFrame(Ŷ=ŷ, X=x)

    lrm = LinearRegressionModel(df, :Ŷ, :X)
    fit!(lrm, atol=1e-7)

    @test all(isapprox.(lrm.argv, [0.1, 0.3], atol=1e-2))
    @test isapprox(predict(lrm, 0.1), 0.1*1 + 0.3*0.1, atol=1e-2)
end
