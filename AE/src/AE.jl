module AE

include("MNIST/MNIST.jl")
using Flux
using CUDA
using JLD2

function get_data(batch_size)
    train_x, train_y = MNIST.traindata(Float32)
    test_x,  test_y  = MNIST.testdata(Float32)

    return (
        Flux.DataLoader((train_x, train_x), batchsize=batch_size, shuffle=true),
        Flux.DataLoader((test_x, test_x), batchsize=batch_size, shuffle=false)
    )
end

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @warn "model updated!"
end

function train(; η₀=1e-4, epoch=500, batch_size=100)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = Chain(
        flatten,
        Dense(28*28, 512, relu),
        Dense(512, 32, relu),

        Dense(32, 512, relu),
        Dense(512, 28*28, relu),
        x -> reshape(x, 28, 28, :)
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    opt = Flux.ADAM(η₀)

    @info "gen data... "
    @time loader_train, loader_test = get_data(batch_size)

    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(@__DIR__, "../model/model_h.jld2"), m)
    end

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    Flux.@epochs epoch @time begin
        Flux.train!(loss, params(m), data, opt)
        validate()
    end
end

function train_dim_reduction(; η₀=1e-4, epoch=500, batch_size=100)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = get_model("model_h")
    m = Chain(
        m[1:3],
        Dense(32, 3),
        Dense(3, 32),
        m[4:6]
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    opt = Flux.ADAM(η₀)

    @info "gen data... "
    @time loader_train, loader_test = get_data(batch_size)

    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(@__DIR__, "../model/model.jld2"), m)
    end

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    Flux.@epochs epoch @time begin
        Flux.train!(loss, params(m), data, opt)
        validate()
    end
end

function get_model(name="model")
    f = jldopen(joinpath(@__DIR__, "../model/$name.jld2"))
    model = f["model"]
    close(f)

    return model
end

end
