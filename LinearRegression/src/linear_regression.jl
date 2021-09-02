using DataFrames

export
    LinearRegressionModel,
    fit!,
    predict

mutable struct LinearRegressionModel{T, S}
    y::Vector{T}
    xs::AbstractArray{S}
    argv::Vector{Float64}
    n::Integer
end

function LinearRegressionModel(
    df::DataFrame,
    label::Symbol,
    features::Vector{Symbol};
    β::Vector{<:Real}=rand(length(features)+1)
)
    n = nrow(df)
    y = df[!, label]
    xs = Matrix(df[!, features])
    if length(β) != length(features) + 1
        throw(DimensionMismatch("Number of features and arguments mismatch."))
    end

    return LinearRegressionModel(y, xs, β, n)
end

function LinearRegressionModel(
    df::DataFrame,
    label::Symbol,
    feature::Symbol;
    β::Vector{<:Real}=rand(2)
)
    n = nrow(df)
    y = df[!, label]
    x = Matrix(df[!, [feature]])
    if length(β) != 2
        throw(DimensionMismatch("Number of features and arguments mismatch."))
    end

    return LinearRegressionModel(y, x, β, n)
end

function y(model::LinearRegressionModel, xs::Matrix{T}) where {T<:Real}
    xs = hcat(ones(T, size(xs, 1)), xs)

    return xs * model.argv
end

residual(model::LinearRegressionModel) = model.y .- y(model, model.xs)

function loss(model::LinearRegressionModel)
    l = sum(x -> 0.5 * x^2, residual(model))

    return l/model.n
end

function ∇L(model::LinearRegressionModel)
    xs = hcat(ones(size(model.xs, 1)), model.xs)

    return vec(sum(-residual(model) .* xs, dims=1))
end

function gradient_descent(model::LinearRegressionModel, η::Real=1e-4, atol::Real=1e-6, show::Bool=false)
    β = model.argv
    while (l = loss(model)) > atol
        show && println("Loss: $l")
        β .-= η .* ∇L(model)
    end

    return β
end

function fit!(model::LinearRegressionModel; method=gradient_descent, η::Real=1e-4, atol::Real=1e-6, show=false)
    β = method(model, η, atol, show)
    model.argv .= β

    return model
end

function predict(model::LinearRegressionModel, xsᵢ::Vector{T}) where {T<:Real}
    xsᵢ = vcat(ones(T, 1), xsᵢ)'

    return xsᵢ * model.argv
end

function predict(model::LinearRegressionModel, xsᵢ::T) where {T<:Real}
    xsᵢ = vcat(ones(T, 1), xsᵢ)'

    return xsᵢ * model.argv
end
