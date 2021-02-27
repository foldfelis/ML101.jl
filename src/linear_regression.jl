export
    LinearRegressionModel,
    fit!,
    predict

struct LinearRegressionModel
    df::DataFrame
    label::Symbol
    features::Vector{Symbol}
    argv::Vector{Float64}
end

"""
    LinearRegressionModel(
        df::DataFrame,
        label::Symbol,
        features::Vector{Symbol};
        argv::Vector{<:Real}
    )

A multipal regression model.

- `df::DataFrame`: DataFrame.
- `label::Symbol`: Specify a valid column name of `df` as a label.
- `features::Vector{Symbol}`: Specify some valid column name of `df` as features.
- `argv::Vector{<:Real}`: Initial arguments.
"""
function LinearRegressionModel(
    df::DataFrame,
    label::Symbol,
    features::Vector{Symbol};
    argv::Vector{<:Real}=rand(length(features)+1)
)
    if length(argv) != length(features) + 1
        throw(DimensionMismatch("Number of features and arguments mismatch."))
    end

    return LinearRegressionModel(df, label, features, argv)
end

"""
    LinearRegressionModel(
        df::DataFrame,
        label::Symbol,
        features::Symbol;
        argv::Vector{<:Real}
    )

A linear regression model.

- `df::DataFrame`: DataFrame.
- `label::Symbol`: Specify a valid column name of `df` as a label.
- `features::Symbol`: Specify a valid column name of `df` as a feature.
- `argv::Vector{<:Real}`: Initial arguments.
"""
function LinearRegressionModel(
    df::DataFrame,
    label::Symbol,
    feature::Symbol;
    argv::Vector{<:Real}=rand(2)
)
    if length(argv) != 2
        throw(DimensionMismatch("Number of features and arguments mismatch."))
    end

    return LinearRegressionModel(df, label, [feature], argv)
end

function g(model::LinearRegressionModel, row_n::Int64)
    xs = collect(model.df[row_n, model.features])
    pushfirst!(xs, 1)
    y = sum(model.argv .* xs)

    return y
end

function loss(model::LinearRegressionModel)
    n = nrow(model.df)
    l = 0
    for i in 1:n
        l += (g(model, i) - model.df[i, model.label])^2
    end

    return l/n
end

function fit!(model::LinearRegressionModel; lr=1e-4, atol::Float64=1e-6, show=false)
    while (l = loss(model)) > atol
        show && println("Loss: $l")
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

function predict(model::LinearRegressionModel, xs::Vector{<:Real})
    pushfirst!(xs, 1)
    y = sum(model.argv .* xs)

    return y
end

function predict(model::LinearRegressionModel, x::Real)
    return predict(model, [x])
end
