### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

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
\end{bmatrix}$

$y = xW_{ComplexFourier}$
"

# ╔═╡ 1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e


# ╔═╡ Cell order:
# ╟─b682a76e-75c6-4e0f-b542-210725501e5b
# ╟─b5ba5d3a-eb07-4d5f-a802-6d85075ed528
# ╟─21b015ea-aa4b-11eb-182b-a7590312f5ba
# ╟─5eeee95e-ce21-4681-a834-9b8e30b935e8
# ╠═1f66b59e-bdcd-4cdb-9f28-3bbf0f7dd08e
