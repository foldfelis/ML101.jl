module NaiveBayes
    using Pluto

    run_notebook() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/naive_bayes.jl"))
end
