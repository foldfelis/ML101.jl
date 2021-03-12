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
    β = rand(length(features))
    return LogisticRegressionModel(y, xs, β, n)
end

function LogisticRegressionModel(df::DataFrame, label::Symbol, feature::Symbol)
    n = nrow(df)
    y = df[!, label]
    x = df[!, feature]
    β = rand()
    return LogisticRegressionModel(y, x, β, n)
end

sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))

z(model::LogisticRegressionModel, xs::Vector{<:Real}) = model.argv' * xs

predict(model::LogisticRegressionModel, xs::Vector{<:Real}) = sigmoid(z(model, xs))

function log_likelyhood(model::LogisticRegressionModel)
    xs = model.xs
    y = model.y

    ll = 0
    for (i, yⁱ) in enumerate(y)
        xsⁱ = xs[i, :]
        ll += yⁱ * log(sigmoid(z(model, xsⁱ)))
        ll += (1 - yⁱ) * log(1 - sigmoid(z(model, xsⁱ)))
    end

    return ll
end

function ∇log_likelyhood(model::LogisticRegressionModel)
    # ∇ll = ∑(yⁱ - σ(zⁱ))xⱼⁱ
    xs = model.xs
    y = model.y

    ∇ll = zeros(length(model.argv))
    for (i, yⁱ) in enumerate(y)
        xsⁱ = xs[i, :]
        ∇ll .+= (yⁱ - sigmoid(z(model, xsⁱ))).* xsⁱ
    end

    return ∇ll
end
