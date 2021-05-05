### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 9fa2be99-1c4a-4e53-8a01-546a3670f19f
begin
    using FFTW
    using Flux
    using Plots
    plotly()
end

# ╔═╡ b682a76e-75c6-4e0f-b542-210725501e5b
html"""<style>
main {
    max-width: 700pt;
}
"""

# ╔═╡ b5ba5d3a-eb07-4d5f-a802-6d85075ed528
md"
# Fourier Net

*“Can neural networks learn the Fourier transform?”*

This notebook is a implementation of [The Fourier transform is a neural network](https://sidsite.com/posts/fourier-nets/).
"

# ╔═╡ 21b015ea-aa4b-11eb-182b-a7590312f5ba
md"
**By the definition of DFT:**

A single amplitude for a singal duration of frequency can be formulated as follow:

$x̂_k = \sum_{n=0}^{N-1} x_n \cdot exp(\frac{-i2\pi k}{N}n)$

$x̂_k =
\begin{bmatrix} x_0 & x_1 & ... & x_{N-1} \end{bmatrix}
\begin{bmatrix}
    exp(0) \\
    exp(\frac{-i 2 \pi k}{N}1) \\
    exp(\frac{-i 2 \pi k}{N}2) \\
    exp(\frac{-i 2 \pi k}{N}3) \\
    exp(\frac{-i 2 \pi k}{N}4) \\
    ... \\
    exp(\frac{-i 2 \pi k}{N}(N-1))
\end{bmatrix}$

Therefore, N of amplitide for N durations of frequency can be illustrated like:

$\begin{bmatrix} x̂_0 \\ ... \\ x̂_{N-1} \end{bmatrix} =
\begin{bmatrix}
    exp(0) & exp(0) & exp(0) & ... & exp(0) \\
    exp(0) & exp(\frac{-i2\pi}{N}1) & exp(\frac{-i2\pi 2}{N}1) & ... & exp(\frac{-i2\pi (N-1)}{N}1) \\
    exp(0) & exp(\frac{-i2\pi}{N}2) & exp(\frac{-i2\pi 2}{N}2) & ... & exp(\frac{-i2\pi (N-1)}{N}2) \\
    exp(0) & exp(\frac{-i2\pi}{N}3) & exp(\frac{-i2\pi 2}{N}3) & ... & exp(\frac{-i2\pi (N-1)}{N}3) \\
    exp(0) & exp(\frac{-i2\pi}{N}4) & exp(\frac{-i2\pi 2}{N}4) & ... & exp(\frac{-i2\pi (N-1)}{N}4) \\
    ... \\
    exp(0) & exp(\frac{-i2\pi}{N}(N-1)) & exp(\frac{-i2\pi 2}{N}(N-1)) & ... & exp(\frac{-i2\pi (N-1)}{N}(N-1))
\end{bmatrix}_{N \times N}^T
\begin{bmatrix} x_0 \\ ... \\ x_{N-1} \end{bmatrix}$

Or in short:

$x̂⃗ = 𝐖_{ComplexFourier}^T x⃗$
"

# ╔═╡ 212ac582-1270-49da-9cd4-078ab695be66
md"
## Generate random signal

$x⃗ = \begin{bmatrix} x_0 \\ ... \\ x_{N-1} \end{bmatrix}$
"

# ╔═╡ 7efa004c-57c4-4b9c-bc74-90252de25218
begin
    len = 128
    x = rand(len)
end;

# ╔═╡ 0d6607c2-0881-40f7-81fd-494d49faa697
md"
## Computing the Fourier transform using Fourier weights
"

# ╔═╡ eeded1aa-3edd-4aa7-b147-9c69e2293267
md"
**Create Fourier weighrts**

$𝐖_{ComplexFourier}^T = \begin{bmatrix}
    exp(0) & exp(0) & exp(0) & ... & exp(0) \\
    exp(0) & exp(\frac{-i2\pi}{N}1) & exp(\frac{-i2\pi 2}{N}1) & ... & exp(\frac{-i2\pi (N-1)}{N}1) \\
    exp(0) & exp(\frac{-i2\pi}{N}2) & exp(\frac{-i2\pi 2}{N}2) & ... & exp(\frac{-i2\pi (N-1)}{N}2) \\
    exp(0) & exp(\frac{-i2\pi}{N}3) & exp(\frac{-i2\pi 2}{N}3) & ... & exp(\frac{-i2\pi (N-1)}{N}3) \\
    exp(0) & exp(\frac{-i2\pi}{N}4) & exp(\frac{-i2\pi 2}{N}4) & ... & exp(\frac{-i2\pi (N-1)}{N}4) \\
    ... \\
    exp(0) & exp(\frac{-i2\pi}{N}(N-1)) & exp(\frac{-i2\pi 2}{N}(N-1)) & ... & exp(\frac{-i2\pi (N-1)}{N}(N-1))
\end{bmatrix}_{N \times N}^T$
"

# ╔═╡ 1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e
function create_𝐰_fourier(len::Integer)
    𝐰 = Matrix{ComplexF64}(undef, len, len)
    for n in 0:len-1, k in 0:len-1
        𝐰[n+1, k+1] = exp(-im*2π*k*n/len)
    end

    return 𝐰'
end;

# ╔═╡ 036016cd-e4f5-40f7-96d9-cac8fdbebd01
md"
**Do the Fourier transform by Fourier weights and FFT**

$x̂⃗ = 𝐖_{ComplexFourier}^T x⃗$

$x̂⃗_{fft} = 𝐹x⃗$
"

# ╔═╡ d5933fdb-b9ed-4475-af43-58dac24a47d5
begin
    # Fourier transform via Fourier weights
    𝐰 = create_𝐰_fourier(len)
    x̂ = 𝐰 * x

    # Fourier transform via fft
    x̂_fft = fft(x)

    # root-mean-sqrt error between those two
    ϵ_x̂ = sqrt(sum(abs.(x̂_fft - x̂).^2)/len)
end;

# ╔═╡ 9d937bef-f325-4299-8c4b-7ae358aeb1c9
md"
Spectrum's RMSE: $ϵ_x̂
"

# ╔═╡ f65c7059-9c2e-4993-a7e6-32cb9a848f23
begin
    plot(title="Spectrum", xlabel="Freq (arb. unit)", ylabel="Amp (arb. unit)")
    plot!(real(x̂_fft), label="FFT")
    plot!(real(x̂), label="Fourier weights")
end

# ╔═╡ 6e200b44-97fb-43e0-8026-aabef11f4f17
md"
**Reconstruct signal by inverse Fourier weights and iFFT**

$x̂_k = \sum_{n=0}^{N-1} x_n \cdot exp(\frac{-i2\pi k}{N}n)$

$x_t = \frac{1}{N} \sum_{k=0}^{N-1} x̂_k \cdot exp(\frac{i2\pi k}{N}n)$
"

# ╔═╡ 45844ffc-9f5c-4952-abde-4ce770ab77f6
function create_𝐰¯¹_fourier(len::Integer)
    𝐰¯¹ = Matrix{ComplexF64}(undef, len, len)
    for n in 0:len-1, k in 0:len-1
        𝐰¯¹[n+1, k+1] = exp(im*2π*k*n/len)
    end

    return (𝐰¯¹/len)'
end;

# ╔═╡ 856e62d3-53c8-49a9-9061-d223dbb92ef1
md"
**Do the inverse Fourier transform by inverse Fourier weights and iFFT**

$x⃗ = 𝐖¯¹_{ComplexFourier}^T x̂⃗$

$x⃗_{fft} = 𝐹¯¹x⃗$
"

# ╔═╡ ac36e1c5-dade-4b0f-974e-2d261ea260f8
begin
    # reconstruct signal via Fourier weights
    𝐰¯¹ = create_𝐰¯¹_fourier(len)
    𝑓¯¹x̂ = real(𝐰¯¹ * x̂)

    # reconstruct signal via ifft
    𝑓¯¹x̂_fft = real(ifft(x̂_fft))

    # root-mean-sqrt error between those two
    ϵ_𝑓¯¹x̂_fft = sqrt(sum(abs.(𝑓¯¹x̂_fft - x).^2)/len)
    ϵ_𝑓¯¹x̂ = sqrt(sum(abs.(𝑓¯¹x̂ - x).^2)/len)
end;

# ╔═╡ 7339df20-651e-42ac-8286-e9696fb72458
md"
Inverse Fourier transform's RMSE:

* Via IFFT: $ϵ_𝑓¯¹x̂_fft
* Via Fourier waights: $ϵ_𝑓¯¹x̂
"

# ╔═╡ e9c246d8-ae77-4236-8ca0-15de7cced221
begin
    plot(title="Signal", xlabel="Time (arb. unit)", ylabel="Amp (arb. unit)", ticks=[], ylims=(0, 3.5))
    plot!(x .+2 , label="Original signal")
    plot!(𝑓¯¹x̂ .+ 1, label="Reconstructed signal (Fourier weights)")
    plot!(𝑓¯¹x̂_fft, label="Reconstructed signal (ifft)")
end

# ╔═╡ 3c03ec6d-8994-4886-9006-3cddbf0ac027
md"
## Learning the Fourier transform via gradient-descent

We can consider the discrete Fourier transform (DFT) to be an artificial neural network: it is a single layer network, with no bias, no activation function, and particular values for the weights. The number of output nodes is equal to the number of frequencies we evaluate.
"

# ╔═╡ 0b4853ce-8dd0-4fe5-881d-cdf5d0289cc8
md"
**Build Fourier Net**
"

# ╔═╡ 1cbd900c-b74b-41c8-a57c-6a6ae644b88c
begin
    mutable struct FourierNet
        𝐰
    end

    Flux.@functor FourierNet

    (m::FourierNet)(x) = m.𝐰 * x

    loss(m::FourierNet, x, x̂) = sum(abs2, x̂ .- m(x)) / len
end;

# ╔═╡ 0abb581e-6835-4dfc-bd01-5b2b6306d242
md"
**Initiating the training process**
"

# ╔═╡ 895ee250-e3ac-4879-b8ee-f1ba8729adaa
begin
    # define the net and loss function
    m = FourierNet(rand(ComplexF64, len, len))
    loss(x, x̂) = loss(m, x, x̂)

    # prepare data
    xs = rand(ComplexF64, len, 50000)
    x̂s = 𝐰 * xs
    data = Flux.Data.DataLoader((xs, x̂s), batchsize=15, shuffle=true)

    # define call-back function for plotting
    training_loss = []
    function cb()
        lossₜ = loss(xs, x̂s)
        push!(training_loss, lossₜ)
    end

    # training process
    train_𝐰!(n, η) = Flux.@epochs n Flux.Optimise.train!(
        loss,
        params(m),
        data,
        Descent(η),
        # cb=Flux.throttle(@show(loss(xs, x̂s)), 1, leading=false, trailing=true)
        cb=Flux.throttle(cb, 1, leading=false, trailing=true)
    )

    # start training
    train_𝐰!(15, 1e-1)
    train_𝐰!(30, 1e-2)
    train_𝐰!(20, 1e-3)
end;

# ╔═╡ 3c1157bd-9780-429a-9cbf-cd5a18d676da
plot(training_loss, title="Loss", xlabel="time (sec)", ylabel="loss", label="training")

# ╔═╡ 9ef928f0-5577-4507-8fbe-193ce6925716
md"
Weights' residual: $(sum(abs, 𝐰 - m.𝐰)/length(𝐰))

Weights' RMSE: $(sqrt(sum(abs2, 𝐰 - m.𝐰)/length(𝐰)))
"

# ╔═╡ 04189afa-6cdd-45f4-8eaa-12140bde3ada
begin
    x̂_grad = m(x)

    plot(title="Spectrum", xlabel="Freq (arb. unit)", ylabel="Amp (arb. unit)")
    plot!(real(x̂_grad), label="Fourier weights by GD")
    plot!(real(x̂_fft), label="FFT")
end

# ╔═╡ Cell order:
# ╟─b682a76e-75c6-4e0f-b542-210725501e5b
# ╟─b5ba5d3a-eb07-4d5f-a802-6d85075ed528
# ╠═9fa2be99-1c4a-4e53-8a01-546a3670f19f
# ╟─21b015ea-aa4b-11eb-182b-a7590312f5ba
# ╟─212ac582-1270-49da-9cd4-078ab695be66
# ╠═7efa004c-57c4-4b9c-bc74-90252de25218
# ╟─0d6607c2-0881-40f7-81fd-494d49faa697
# ╟─eeded1aa-3edd-4aa7-b147-9c69e2293267
# ╠═1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e
# ╟─036016cd-e4f5-40f7-96d9-cac8fdbebd01
# ╠═d5933fdb-b9ed-4475-af43-58dac24a47d5
# ╟─9d937bef-f325-4299-8c4b-7ae358aeb1c9
# ╠═f65c7059-9c2e-4993-a7e6-32cb9a848f23
# ╟─6e200b44-97fb-43e0-8026-aabef11f4f17
# ╠═45844ffc-9f5c-4952-abde-4ce770ab77f6
# ╟─856e62d3-53c8-49a9-9061-d223dbb92ef1
# ╠═ac36e1c5-dade-4b0f-974e-2d261ea260f8
# ╟─7339df20-651e-42ac-8286-e9696fb72458
# ╠═e9c246d8-ae77-4236-8ca0-15de7cced221
# ╟─3c03ec6d-8994-4886-9006-3cddbf0ac027
# ╟─0b4853ce-8dd0-4fe5-881d-cdf5d0289cc8
# ╠═1cbd900c-b74b-41c8-a57c-6a6ae644b88c
# ╟─0abb581e-6835-4dfc-bd01-5b2b6306d242
# ╠═895ee250-e3ac-4879-b8ee-f1ba8729adaa
# ╠═3c1157bd-9780-429a-9cbf-cd5a18d676da
# ╟─9ef928f0-5577-4507-8fbe-193ce6925716
# ╠═04189afa-6cdd-45f4-8eaa-12140bde3ada
