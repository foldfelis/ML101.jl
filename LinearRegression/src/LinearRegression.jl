module LinearRegression
    using Pluto

    include("linear_regression.jl")

    run_notebook() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/ice_cream_vs_whiskey.jl"))
end
