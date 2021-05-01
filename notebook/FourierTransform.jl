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

# ╔═╡ 21b015ea-aa4b-11eb-182b-a7590312f5ba
md"
$y_k = 
\begin{bmatrix} x_0 & x_1 & ... & x_{N-1} \end{bmatrix} 
\begin{bmatrix} 
	exp(0) \\ 
	exp(\frac{-i 2 \pi k}{N}) \\ 
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

# ╔═╡ Cell order:
# ╠═b682a76e-75c6-4e0f-b542-210725501e5b
# ╠═21b015ea-aa4b-11eb-182b-a7590312f5ba
# ╠═5eeee95e-ce21-4681-a834-9b8e30b935e8
