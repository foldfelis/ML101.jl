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

function g(model::LinearRegressionModel, row_n::Int64)
    xs = collect(model.df[row_n, model.features])
    pushfirst!(xs, 1)
    y = sum(model.argv .* xs)

    return y
end

function loss(model::LinearRegressionModel)
    l = 0
    for i in 1:nrow(model.df)
        l += (g(model, i) - model.df[i, model.label])^2
    end

    return l
end

function fit!(model::LinearRegressionModel; lr=1e-4, atol::Float64=1e-6)
    while loss(model) > atol
        dl_da = 0
        dl_db = zeros(length(model.features))
        for i in 1:nrow(model.df)
            # intersection
            dl_da += g(model, i) - model.df[i, model.label]

            # sloaps
            for (j, f) in enumerate(model.features)
                dl_db[j] += (g(model, i) - model.df[i, model.label]) * model.df[i, f]
            end
        end

        # intersection
        model.argv[1] -= lr * 2 * dl_da
        # slopes
        for j in 2:length(model.argv)
            model.argv[j] -= lr * 2 * dl_db[j-1]
        end
    end
end
