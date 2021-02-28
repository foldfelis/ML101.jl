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

# y = g(x) = a + b1 x1 + b2 x2 ...
function g(model::LinearRegressionModel, row_n::Int64)
    xs = collect(model.df[row_n, model.features])
    pushfirst!(xs, 1)
    y = xs' * model.argv

    return y
end

function ĝ(model::LinearRegressionModel, row_n::Int64)
    return model.df[row_n, model.label]
end

function loss(model::LinearRegressionModel)
    n = nrow(model.df)
    l = 0
    for i in 1:n
        l += (g(model, i) - ĝ(model, i))^2
    end

    return l/n
end

function ∇loss(model::LinearRegressionModel)
    dl_da = 0
    dl_db = zeros(length(model.features))
    for i in 1:nrow(model.df)
        # intercept
        dl_da += g(model, i) - ĝ(model, i)

        # slopes
        for (j, f) in enumerate(model.features)
            dl_db[j] += (g(model, i) - ĝ(model, i)) * model.df[i, f]
        end
    end

    # merge to one vector
    return 2 .* pushfirst!(dl_db, dl_da)
end

function gradient_descent(model::LinearRegressionModel, ∇loss, η::Real, atol::Real, show=false)
    while (l = loss(model)) > atol
        show && println("Loss: $l")

        dl_vec = ∇loss(model)
        for i in 1:length(model.argv)
            model.argv[i] -= η * dl_vec[i]
        end
    end
end

function fit!(model::LinearRegressionModel; η::Real=1e-4, atol::Real=1e-6, show=false)
    gradient_descent(model, ∇loss, η, atol, show)
end

function predict(model::LinearRegressionModel, xs::Vector{<:Real})
    pushfirst!(xs, 1)
    y = xs' * model.argv

    return y
end

function predict(model::LinearRegressionModel, x::Real)
    return predict(model, [x])
end
