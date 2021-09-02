module LearningFourierTransform
    using Pluto

    run_notebook() = Pluto.run(notebook=joinpath(@__DIR__, "../notebook/learning_fourier_transform.jl"))
end
