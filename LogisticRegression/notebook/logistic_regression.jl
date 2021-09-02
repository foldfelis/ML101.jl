### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ c2d0d93c-8d12-4b58-852b-10795ff1f8db
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ c74a1698-83dd-11eb-1bd2-c145ea201500
begin
	using DataFrames
	using LogisticRegression

	using Plots
	using StatsPlots
	gr()
end;

# ╔═╡ 43f669b2-8596-11eb-2e12-efdef4ac0e29
md"
# Demo of Logistic Regression

JingYu Nng
"

# ╔═╡ 135d298a-83de-11eb-1f50-83e4261f2bdf
md"
### Generate mock data
"

# ╔═╡ f374b00a-83dd-11eb-0f93-612e3ed66e75
begin
	n = 100
    β = [1., 1.]
    df = DataFrame(
        X₁=vcat(randn(n).-5, randn(n).+10),
        X₂=vcat(randn(n).-8, randn(n).+16),
        Y=vcat(ones(n), zeros(n))
    )
end

# ╔═╡ 2b9fc1a6-83de-11eb-3863-19df8f1b6252
md"
### Initial model
"

# ╔═╡ 25ae7788-83de-11eb-27be-2d51a3c90c8c
begin
	lrm = LogisticRegressionModel(df, :Y, [:X₁, :X₂])
end;

# ╔═╡ 7a44b194-83df-11eb-2de4-7f16b19d1239
begin
	const C_GRAD = cgrad([
		RGBA(53/255, 157/255, 219/255, 1),
		RGBA(240/255, 240/255, 240/255, 1),
		RGBA(219/255, 64/255, 68/255, 1)
	])

	function plot_model(model::LogisticRegressionModel, df::DataFrame; margin=3)
		lim = maximum(abs.(lrm.xs))
		x = -lim-margin:0.1:lim+margin
		heatmap(
			x, x, (x1, x2)->predict(lrm, [x1, x2]),
			title="Logistic Regression Model",
			xlabel="X₁",
			ylabel="X₂",
			legend=:topleft,
			clim=(0, 1),
			c=C_GRAD,
		)
		@df df[df.Y .== 0, :] scatter!(:X₁ , :X₂, color=C_GRAD[1], label="False")
		@df df[df.Y .== 1, :] scatter!(:X₁ , :X₂, color=C_GRAD[3], label="True")
	end
end;

# ╔═╡ da6bb3a2-83de-11eb-3a43-55dbeaabac48
md"
### Model before training
"

# ╔═╡ e78abf10-83de-11eb-29ca-c72ce80b5877
plot_model(lrm, df)

# ╔═╡ 31343290-83df-11eb-0c9f-114c43fd1c8d
md"
### Training
"

# ╔═╡ 37c5a346-83df-11eb-06f8-9b549328b45f
fit!(lrm, η=1e-2, atol=-1e-4);

# ╔═╡ 4242583c-83df-11eb-08b7-752569c5bbe8
md"
### Model after training
"

# ╔═╡ 8a450b0c-83df-11eb-34b9-4351fb9d772c
plot_model(lrm, df)

# ╔═╡ Cell order:
# ╟─43f669b2-8596-11eb-2e12-efdef4ac0e29
# ╟─c2d0d93c-8d12-4b58-852b-10795ff1f8db
# ╠═c74a1698-83dd-11eb-1bd2-c145ea201500
# ╟─135d298a-83de-11eb-1f50-83e4261f2bdf
# ╠═f374b00a-83dd-11eb-0f93-612e3ed66e75
# ╟─2b9fc1a6-83de-11eb-3863-19df8f1b6252
# ╠═25ae7788-83de-11eb-27be-2d51a3c90c8c
# ╟─7a44b194-83df-11eb-2de4-7f16b19d1239
# ╟─da6bb3a2-83de-11eb-3a43-55dbeaabac48
# ╠═e78abf10-83de-11eb-29ca-c72ce80b5877
# ╟─31343290-83df-11eb-0c9f-114c43fd1c8d
# ╠═37c5a346-83df-11eb-06f8-9b549328b45f
# ╟─4242583c-83df-11eb-08b7-752569c5bbe8
# ╠═8a450b0c-83df-11eb-34b9-4351fb9d772c
