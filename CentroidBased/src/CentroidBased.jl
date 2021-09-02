module CentroidBased
    using Pluto

    run_notebook_k_means_pp() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/k_means_pp.jl"))
    run_notebook_k_medoids() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/k_medoids.jl"))
    run_notebook_medoid_median_mode() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/medoid_median_mode.jl"))
end
