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

function loss(model::LinearRegressionModel)
    l = 0
    for i in 1:nrow(model.df)
        l += (g(model, collect(model.df[i, model.features])) - model.df[i, model.label])^2
    end

    return l
end

# function fit!(model::LinearRegressionModel; lr=1e-3, atol::Float64=1e-6)
#     argv = model.argv

#     argv[1] = ar
# end
