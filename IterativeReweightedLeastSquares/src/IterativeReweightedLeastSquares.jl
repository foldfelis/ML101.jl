module IterativeReweightedLeastSquares
    using Pluto

    run_notebook() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/irls.jl"))
end
