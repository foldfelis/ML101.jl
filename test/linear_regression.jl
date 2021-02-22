@testset "linear_regression.jl function g" begin
    # Ŷ = 0.3 x₁ + 0.6 X₂ + 0.9 x₃
    x₁ = collect(1:10)
    x₂ = collect(11:20)
    x₃ = collect(21:30)
    ŷ = 0.1 .+ 0.3x₁ .+ 0.5x₂ .+ 0.7x₃

    df = DataFrame(Ŷ=ŷ, X₁=x₁, X₂=x₂, X₃=x₃)

    lrm = LinearRegressionModel(df, :Ŷ, [:X₁, :X₂, :X₃], [0.1, 0.3, 0.5, 0.7])

    @test ML101.g(lrm, collect(df[1, [:X₁, :X₂, :X₃]])) == 0.1 + 0.3*1 + 0.5*11 + 0.7*21
end

@testset "linear_regression.jl" begin

    # Ŷ = 0.3 x₁ + 0.6 X₂ + 0.9 x₃
    x₁ = rand(10)
    x₂ = rand(10)
    x₃ = rand(10)
    ŷ = 0.1 .+ 0.3x₁ .+ 0.5x₂ .+ 0.7x₃

    df = DataFrame(Ŷ=ŷ, X₁=x₁, X₂=x₂, X₃=x₃)

    lrm = LinearRegressionModel(df, label=:Ŷ, features=[:X₁, :X₂, :X₃])

end
