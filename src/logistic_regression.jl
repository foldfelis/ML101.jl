export
    LogisticRegressionModel,
    fit!,
    predict

mutable struct LogisticRegressionModel{T,S}
    y::Vector{T}
    xs::AbstractArray{S}
    argv::Vector{Float64}
    n::Integer
end

function LogisticRegressionModel(df::DataFrame, label::Symbol, features::Vector{Symbol})
    n = nrow(df)
    y = df[!, label]
    xs = Matrix(df[!, features])
    β = rand(length(features)+1)
    return LogisticRegressionModel(y, xs, β, n)
end

function LogisticRegressionModel(df::DataFrame, label::Symbol, feature::Symbol)
    n = nrow(df)
    y = df[!, label]
    xs = df[!, feature]
    β = rand(2)
    return LogisticRegressionModel(y, xs, β, n)
end

function sigmoid(z::Real)
    return 1.0 / (1.0 + exp(-z))
end

function predict(model::LogisticRegressionModel, x::Real)
    return sigmoid(model.argv * x)[1]
end

function predict(model::LogisticRegressionModel, xs::Vector{Real})
    return sigmoid(model.argv' * xs)
end
