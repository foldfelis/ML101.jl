module LogisticRegression
    using Pluto

    include("logistic_regression.jl")

    run_notebook_logistic_regression() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/logistic_regression.jl"))
    run_notebook_candy_contain_chocolate() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/candy_contain_chocolate.jl"))
end
