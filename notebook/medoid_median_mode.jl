### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 200b3ca5-0648-489d-b10d-62a851cfdb4a
begin
	using Plots
	plotly()
end

# ╔═╡ aecba716-d27b-11eb-10e0-f529b2367d05
data = vcat(randn(400, 2).-0.5, 2randn(201, 2).+1);

# ╔═╡ 312a2deb-39fb-4c26-b73a-e842f95f245d
begin
	signal = "⚡"
	scatter(data[:, 1], data[:, 2], label="data")
end

# ╔═╡ 51d14132-f210-4982-87b0-9626bd185ae4
begin
	signal
	
	mean = [sum(data[:, 1])/601, sum(data[:, 2])/601]
	scatter!([mean[1]], [mean[2]], label="mean", markersize=10)
	
	medoid_i = argmin(
		[sum(sum(abs.(data[j, :]-data[i, :])) for j in 1:601) for i in 601]
	)
	scatter!([data[1, medoid_i]], [data[2, medoid_i]], label="medoid", markersize=10)
	
	median = sort(data, dims=1)[301, :]
	scatter!([median[1]], [median[2]], label="median", markersize=10)
	
	# mode = maximum(density(data))
end

# ╔═╡ Cell order:
# ╠═200b3ca5-0648-489d-b10d-62a851cfdb4a
# ╠═aecba716-d27b-11eb-10e0-f529b2367d05
# ╟─312a2deb-39fb-4c26-b73a-e842f95f245d
# ╟─51d14132-f210-4982-87b0-9626bd185ae4
