### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 9fb05e18-3480-11ec-372e-f36b56f00c89
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 99f842de-080e-43b8-9d44-63a4dff4a7f7
begin
	using AE
	using Plots
	plotly()
end

# ╔═╡ 390a55c6-0846-4588-9686-05ebfb6d5e0b
data, label = AE.get_test_data_and_label();

# ╔═╡ 81184495-8fd8-491e-a761-ce371bb05adb
m = AE.get_model("model_h")

# ╔═╡ 509bbbb6-d1e8-4593-b829-c8d670d885a2
function plot_data(i::Integer, size=(580, 280))
	p1 = heatmap(
		data[:, end:-1:1, i]', 
		color=:coolwarm, clim=(-1, 1),
		axis=false, ticks=[], colorbar=false, xlabel="Ground Truth",
		size=(280, 280)
	)
	p2 = heatmap(
		m(data[:, :, i:i])[:, end:-1:1, 1]', 
		color=:coolwarm, clim=(-1, 1),
		axis=false, ticks=[], colorbar=false, xlabel="Inferenced data",
		size=(280, 280)
	)
	
	return plot(p1, p2, size=size)
end;

# ╔═╡ 1f31a397-521c-419e-be91-3457e3d5e429
plot_data(4, (290, 140))

# ╔═╡ aa46c772-e624-46cf-af64-1821e2e1acea
plot_data(3, (290, 140))

# ╔═╡ b0ddc77d-13ca-493c-ab37-e9d9e90683ef
plot_data(2, (290, 140))

# ╔═╡ 8fd27e2b-e4c1-4708-bf3c-92560ba1da4f
plot_data(19, (290, 140))

# ╔═╡ 7b818fe7-0bcf-4199-87c8-3f1339764fd6
plot_data(5, (290, 140))

# ╔═╡ c6700ed5-8ca7-46bb-835f-ae1ef9495651
plot_data(9, (290, 140))

# ╔═╡ 38ab2928-1fd8-46a4-a82e-79c29857743e
plot_data(12, (290, 140))

# ╔═╡ 11bafa2b-7396-4b19-bf0e-c58dacada767
plot_data(1, (290, 140))

# ╔═╡ 5f5b90e6-7695-46ce-b598-cbb3e37be1b9
plot_data(62, (290, 140))

# ╔═╡ df959e23-199f-4410-9475-a77ea77c24f6
plot_data(8, (290, 140))

# ╔═╡ d17b1e5e-390d-4b6a-addd-4f2096646113
m_dim_reduced = AE.get_model()

# ╔═╡ 32538eb6-57af-4582-9854-a2089ae934fb
embed = m_dim_reduced[1:2]

# ╔═╡ 15d6fc1a-e489-41a4-adff-a276a39f8415
embeddiing = vcat(embed(data), reshape(label, 1, :))

# ╔═╡ e39a081f-27b5-4934-9259-1d87c9077d35
scatter(
	embeddiing[1, :], embeddiing[2, :], embeddiing[3, :], 
	group=Int.(embeddiing[end, :]),
	markersize=2, size=(650, 850), legend=:right
)

# ╔═╡ Cell order:
# ╟─9fb05e18-3480-11ec-372e-f36b56f00c89
# ╠═99f842de-080e-43b8-9d44-63a4dff4a7f7
# ╠═390a55c6-0846-4588-9686-05ebfb6d5e0b
# ╠═81184495-8fd8-491e-a761-ce371bb05adb
# ╟─509bbbb6-d1e8-4593-b829-c8d670d885a2
# ╟─1f31a397-521c-419e-be91-3457e3d5e429
# ╟─aa46c772-e624-46cf-af64-1821e2e1acea
# ╟─b0ddc77d-13ca-493c-ab37-e9d9e90683ef
# ╟─8fd27e2b-e4c1-4708-bf3c-92560ba1da4f
# ╟─7b818fe7-0bcf-4199-87c8-3f1339764fd6
# ╟─c6700ed5-8ca7-46bb-835f-ae1ef9495651
# ╟─38ab2928-1fd8-46a4-a82e-79c29857743e
# ╟─11bafa2b-7396-4b19-bf0e-c58dacada767
# ╟─5f5b90e6-7695-46ce-b598-cbb3e37be1b9
# ╟─df959e23-199f-4410-9475-a77ea77c24f6
# ╠═d17b1e5e-390d-4b6a-addd-4f2096646113
# ╠═32538eb6-57af-4582-9854-a2089ae934fb
# ╠═15d6fc1a-e489-41a4-adff-a276a39f8415
# ╟─e39a081f-27b5-4934-9259-1d87c9077d35
