### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 9daa72fd-1145-4733-a21e-bd209367380a
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 947fbdc9-3c48-44f3-b4d3-183f70e2194e
using LinearAlgebra

# ╔═╡ a465636e-bc12-4e8f-9db2-1a20ea771504
begin
	𝐱 = rand(100, 3)
	𝐱 = hcat(ones(size(𝐱, 1)), 𝐱)
	y = 𝐱 * [1, 2, 3, 4] + 1e-3randn(size(𝐱, 1))
end;

# ╔═╡ 833c56c2-d2ab-4f94-968b-b5771c9fa945
function irls(𝐱, y; p=2, n_iter=100)
	w = ones(size(𝐱, 1))
	𝐰 = diagm(w)
	β = inv(𝐱' * 𝐰 * 𝐱) * 𝐱' * 𝐰 * y
	
	if p == 2
		return β
	end
	
	for _ in 1:n_iter
		w = abs.(y - 𝐱 * β).^(p-2)
		𝐰 = diagm(w)
		β = inv(𝐱' * 𝐰 * 𝐱) * 𝐱' * 𝐰 * y
	end
	
	return β
end

# ╔═╡ 2c663777-b6e3-42b2-a279-d7b42dd27d70
irls(𝐱, y, p=4, n_iter=10)

# ╔═╡ Cell order:
# ╟─9daa72fd-1145-4733-a21e-bd209367380a
# ╠═947fbdc9-3c48-44f3-b4d3-183f70e2194e
# ╠═a465636e-bc12-4e8f-9db2-1a20ea771504
# ╠═833c56c2-d2ab-4f94-968b-b5771c9fa945
# ╠═2c663777-b6e3-42b2-a279-d7b42dd27d70
