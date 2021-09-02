module PCA
    using Pluto

    run_notebook() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/pca.jl"))
end
