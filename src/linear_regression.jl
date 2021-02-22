export
    LinearRegressionModel,
    fit!

mutable struct LinearRegressionModel
    df::DataFrame
    label::Symbol
    features::Vector{Symbol}
    argv::Vector{Float64}
end

function LinearRegressionModel(df::DataFrame; label::Symbol, features::Vector{Symbol})
    return LinearRegressionModel(
        df,
        label,
        features,
        rand(length(features) + 1)
    )
end

function g(model::LinearRegressionModel, xs::Vector{<:Real})
    argv = model.argv

    y = argv[1]
    for (a, x) in zip(argv[2:end], xs)
        y += a * x
    end

    return y
end

function fit!(model::LinearRegressionModel; atol::Float64=1e-6)

end
