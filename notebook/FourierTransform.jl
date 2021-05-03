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
$y_k = \sum_{n=0}^{N-1} x_n \cdot exp(\frac{-i2\pi k}{N}n)$

$y_k = 
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
"

# ╔═╡ 5eeee95e-ce21-4681-a834-9b8e30b935e8
md"
$\begin{bmatrix} y_0\ ...\ y_{N-1} \end{bmatrix} = 
\begin{bmatrix} x_0\ ...\ x_{N-1} \end{bmatrix} 
\begin{bmatrix} 
	exp(0) & exp(0) & exp(0) & ... & exp(0) \\ 
	exp(0) & exp(\frac{-i2\pi}{N}1) & exp(\frac{-i2\pi 2}{N}1) & ... & exp(\frac{-i2\pi (N-1)}{N}1) \\ 
	exp(0) & exp(\frac{-i2\pi}{N}2) & exp(\frac{-i2\pi 2}{N}2) & ... & exp(\frac{-i2\pi (N-1)}{N}2) \\
	exp(0) & exp(\frac{-i2\pi}{N}3) & exp(\frac{-i2\pi 2}{N}3) & ... & exp(\frac{-i2\pi (N-1)}{N}3) \\
	exp(0) & exp(\frac{-i2\pi}{N}4) & exp(\frac{-i2\pi 2}{N}4) & ... & exp(\frac{-i2\pi (N-1)}{N}4) \\
	... \\
	exp(0) & exp(\frac{-i2\pi}{N}(N-1)) & exp(\frac{-i2\pi 2}{N}(N-1)) & ... & exp(\frac{-i2\pi (N-1)}{N}(N-1))
\end{bmatrix}_{N \times N}$

$y = xW_{ComplexFourier}$
"

# ╔═╡ 0d6607c2-0881-40f7-81fd-494d49faa697
md"
## Computing the Fourier transform using Fourier weights
"

# ╔═╡ 1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e
function create_𝐰_fourier(len::Integer)
	𝐰 = Matrix{ComplexF64}(undef, len, len)
	for n in 0:len-1, k in 0:len-1
		𝐰[n+1, k+1] = exp(-im*2π*k*n/len)
	end
	
	return 𝐰
end;

# ╔═╡ 45844ffc-9f5c-4952-abde-4ce770ab77f6
function create_𝐰¯¹_fourier(len::Integer)
	𝐰¯¹ = Matrix{ComplexF64}(undef, len, len)
	for n in 0:len-1, k in 0:len-1
		𝐰¯¹[n+1, k+1] = exp(im*2π*k*n/len)
	end
	
	return 𝐰¯¹/len
end;

# ╔═╡ 212ac582-1270-49da-9cd4-078ab695be66
md"
Generate random signal

$x = \begin{bmatrix} x_0\ ...\ x_{N-1} \end{bmatrix}$
"

# ╔═╡ 7efa004c-57c4-4b9c-bc74-90252de25218
begin
	len = 64
	x = rand(len)'
end;

# ╔═╡ 036016cd-e4f5-40f7-96d9-cac8fdbebd01
md"
Do the Fourier transform by Fourier seights and FFT

$x̂ = Fx$
"

# ╔═╡ d5933fdb-b9ed-4475-af43-58dac24a47d5
begin	
	# Fourier transform via Fourier weights
	𝐰 = create_𝐰_fourier(len)
	x̂ = x * 𝐰
	
	# Fourier transform via fft
	x̂_fft = fft(x)

	# root-mean-sqrt error between those two
	ϵ = sqrt(sum(abs.(x̂_fft - x̂).^2)/len)
end;

# ╔═╡ ac36e1c5-dade-4b0f-974e-2d261ea260f8
begin
	# reconstruct signal via Fourier weights
	𝐰¯¹ = create_𝐰¯¹_fourier(len)
	𝑓¯¹x̂ = real(x̂ * 𝐰¯¹)'
	
	# reconstruct signal via ifft
	𝑓¯¹x̂_fft = real(ifft(x̂_fft))'
end;

# ╔═╡ e9c246d8-ae77-4236-8ca0-15de7cced221
begin
	plot(title="Signal", xlabel="Time (arb. unit)", ylabel="Amp (arb. unit)", ticks=[], ylims=(-2.5, 2))
	plot!(x', label="Original signal")
	plot!(𝑓¯¹x̂ .- 1, label="Reconstructed signal (Fourier weights)")
	plot!(𝑓¯¹x̂_fft .- 2, label="Reconstructed signal (ifft)")
end

# ╔═╡ 3c03ec6d-8994-4886-9006-3cddbf0ac027
md"
## Learning the Fourier transform via gradient-descent
"

# ╔═╡ Cell order:
# ╟─b682a76e-75c6-4e0f-b542-210725501e5b
# ╟─b5ba5d3a-eb07-4d5f-a802-6d85075ed528
# ╠═9fa2be99-1c4a-4e53-8a01-546a3670f19f
# ╟─21b015ea-aa4b-11eb-182b-a7590312f5ba
# ╟─5eeee95e-ce21-4681-a834-9b8e30b935e8
# ╟─0d6607c2-0881-40f7-81fd-494d49faa697
# ╠═1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e
# ╠═45844ffc-9f5c-4952-abde-4ce770ab77f6
# ╟─212ac582-1270-49da-9cd4-078ab695be66
# ╠═7efa004c-57c4-4b9c-bc74-90252de25218
# ╟─036016cd-e4f5-40f7-96d9-cac8fdbebd01
# ╠═d5933fdb-b9ed-4475-af43-58dac24a47d5
# ╠═ac36e1c5-dade-4b0f-974e-2d261ea260f8
# ╠═e9c246d8-ae77-4236-8ca0-15de7cced221
# ╟─3c03ec6d-8994-4886-9006-3cddbf0ac027
