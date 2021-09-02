### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 8c102f1e-e978-4ed3-8620-830c46bb566f
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 0ae615d5-739f-456f-be76-d7d3307577bb
begin
	using LinearAlgebra
	using StatsBase
	using CSV
	using DataFrames
	using RDatasets
	using Plots
	using StatsPlots
	plotly(size=(680, 500))
end

# ╔═╡ 9bbf3d20-0b30-11ec-1917-ebba9c2ee5f0
md"
# Principal Component Analysis

JingYu Ning

**PCA** is a promissing **decorrelating dimension reduction.**
"

# ╔═╡ d67717f9-b5d6-4f84-ad80-31fe0c93b014
md"
## Let's start from a simple example

dataset: [Height and shoe size](https://osf.io/ja9dw/)

It is said that the length of ones foot is corelated with ones height...
"

# ╔═╡ a4d300fb-f712-4045-a8b8-7bf3b4b55ecc
md"
### Show data
"

# ╔═╡ 820f64a0-287d-40fa-b5f5-2cb35dc91658
SIG = "⚡";

# ╔═╡ 98d1834d-f61e-4b28-a6bc-9dec42a1bff8
begin
	SIG
	raw = CSV.read(joinpath(@__DIR__, "../data/HeightAndShoeSize.csv"), DataFrame)
	raw = raw[!, 2:end]
	data = copy(raw)
end

# ╔═╡ 53400761-545e-4f2a-ae75-80694a43dd92
@df data scatter(
	:height, :shoe_size, group=:sex,
	xlabel="height", ylabel="shoe size",
	legend=:topleft
)

# ╔═╡ 033440c0-efc2-42c4-918a-e0d966e27d3f
md"
### 1. Move the data to the origin
"

# ╔═╡ b3647ffa-d904-4bee-9327-6dec393702e5
begin
	for i in 2:3
		data[!, i] .-= mean(data[!, i])
		raw[!, i] .-= mean(raw[!, i])
	end
	@df data scatter(
		:height, :shoe_size, group=:sex,
		xlabel="height", ylabel="shoe size",
		legend=:topleft,
		xlim=(-40, 40), ylim=(-40, 40)
	)
end

# ╔═╡ 4f771fd7-84be-43f9-a9b2-78d40db4f91c
md"
### 2. SVD

$X = U \Sigma V^\dagger$
"

# ╔═╡ 6b6365a7-486a-4533-bd1e-d6c5726f3b35
u, s, v = svd(Matrix(data[!, 2:3])')

# ╔═╡ ddfc6838-cf88-4aa2-a95c-5d4c95a487ae
md"
### 3. Principal components

Choose the orthogonal basis `u` as the **principal components**
"

# ╔═╡ 66806d5b-55b7-4a4f-b592-2fe6304a7795
begin
	scale = 10
	p = @df data scatter(
		:height, :shoe_size, group=:sex,
		xlabel="height", ylabel="shoe size",
		legend=:topleft,
		xlim=(-40, 40), ylim=(-40, 40)
	)
	for i in 1:size(u, 1)
		pci = scale * u[:, i]
		plot!(p, [0, pci[1]], [0, pci[2]], label="pc$i", lw=5)
	end
	p
end

# ╔═╡ de3026a5-109e-458c-9e38-8e6ab43a81c6
md"
### 4. Apply the basis

$U^T X$
"

# ╔═╡ 5530ddd4-df5e-4308-9dfd-492f68db9f94
begin
	data[!, 2:3] .= (u' * Matrix(data[!, 2:3])')'
	@df data scatter(
		:height, :shoe_size, group=:sex,
		xlabel="pc1", ylabel="pc2",
		legend=:topleft,
		xlim=(-40, 40), ylim=(-40, 40)
	)
end

# ╔═╡ c7490656-5f5f-41a5-8a56-e3341880d855
md"
### 5. Rescale

$\Sigma^{-1} U^T X$
"

# ╔═╡ 89767b3c-63eb-4bdb-9f96-2289954908fa
begin
	data[!, 2:3] .= (inv(diagm(s)) * Matrix(data[!, 2:3])')'
	@df data scatter(
		:height, :shoe_size, group=:sex,
		xlim=(-0.5, 0.5), ylim=(-0.5, 0.5)
	)
end

# ╔═╡ 4ae5b880-1c74-4d37-b1eb-6f6f847b39a0
md"
$I = \Sigma^{-1} U^T X V$
"

# ╔═╡ 6a3f7794-3776-48cf-b188-ee892078c148
Matrix(data[!, 2:3])' * v

# ╔═╡ edcc9e1c-9cc0-4ead-ae6f-4c68bc509470
md"
### 6. Dimension reduction

We want to map those data into lower dimentions with less intormance loss.

Here, $U$ is a basis in the original high dimentional space. We will have least informance loss if project data into one of the axis of $U$, while have most loss if into another.

$U = [v_1, v_2]$
"

# ╔═╡ 18c0c325-c107-42e8-84a3-587580cc5772
md"
By choosing a new basis, we can therefore drop one dimention with little information lossed.
"

# ╔═╡ 88526e35-67c0-430a-a6ef-6454cdab7d2d
begin
	plot(size=(680, 200), yticks=[])
	scatter!((u'*Matrix(raw[!, 2:3])')[1, :], zeros(nrow(raw)), label="pc1")
	scatter!((u'*Matrix(raw[!, 2:3])')[2, :], ones(nrow(raw)), label="pc2")
end

# ╔═╡ 4d5c2e27-8a2c-486b-97bd-65ce4d108322
md"
And the singular value is the information of varience.
"

# ╔═╡ 5312759e-6c75-411b-b28f-9de6f6bfa65b
begin
	plot(size=(680, 200), yticks=[])
	scatter!(
		(inv(diagm(s))*u'*Matrix(raw[!, 2:3])')[1, :], zeros(nrow(raw)),
		label="pc1"
	)
	scatter!(
		(inv(diagm(s))*u'*Matrix(raw[!, 2:3])')[2, :], ones(nrow(raw)),
		label="pc2"
	)
end

# ╔═╡ d5fe909b-784a-443e-9e2e-c00c9e8cd611
md"
We constract a mapping that maps the data into `Z` space

$W = \Sigma^{-1}U^T$
"

# ╔═╡ e3e19d9e-767a-486b-8626-108c5f70a63b
w = inv(diagm(s)) * u'

# ╔═╡ d315c4c8-9945-409a-90b5-f08369665b25
md"The mapping is roughly the overall picture of the data."

# ╔═╡ 9658209d-57b5-4d68-a031-e15711d8cc92
md"
## Iris dataset
"

# ╔═╡ 7475abbd-6fdb-423f-8981-be4461df6648
iris = dataset("datasets", "iris")

# ╔═╡ 663406ed-49d2-439b-8272-69c0b29546da
begin
    cols = names(iris)

    function gen_pots()
        plots = Matrix{Plots.Plot}(undef, 4, 4)
        for i in 1:4, j in 1:4
            if i == j
                plots[i, j] = @df(iris, histogram(cols(i)))
            elseif i < j
                plots[i, j] = @df(iris, histogram2d(
                    cols(i), cols(j),
                    nbins=20, color=:coolwarm, colorbar=false
                ))
            else
                plots[i, j] =  @df(iris, scatter(cols(i), cols(j)))
            end

            (i==1) && (plots[i, j] = plot!(ylabel=cols[j])) # add x label
            (j==4) && (plots[i, j] = plot!(xlabel=cols[i])) # add y label
        end

        return plots
    end
end

# ╔═╡ f79ca361-a2ca-4e0e-b920-36936d01fe8b
plot(gen_pots()..., layout=(4, 4), legend=false)

# ╔═╡ 7a284458-83bb-48fb-863e-ff385a3ba27d
md"
### 1. Move the data to the origin
"

# ╔═╡ da5d583f-b9d8-4c96-892c-c3a018d83890
begin
	for i in 1:(ncol(iris)-1)
		iris[!, i] .-= mean(iris[!, i])
	end
	plot(gen_pots()..., layout=(4, 4), legend=false)
end

# ╔═╡ 221d3ebf-e734-4a7d-8811-d8874311aa21
md"
### 2. SVD
"

# ╔═╡ 58db4367-514b-4ce8-b994-a13d73af93e9
U, S, V = svd(Matrix(iris[!, 1:(ncol(iris)-1)])')

# ╔═╡ 4a0d265d-9c91-4095-899b-20e4c6765fbf
md"
### 3. Construct the mapping

$W = \Sigma^{-1}U^T$
"

# ╔═╡ 1227c477-2fa5-4929-b64b-aeaf5745c52d
W = inv(diagm(S)) * U'

# ╔═╡ c1b885d9-bb2b-486f-ad22-219e4ede6025
begin
	iris[!, 1:(ncol(iris)-1)] .= (W * Matrix(iris[!, 1:(ncol(iris)-1)])')'
	plot(gen_pots()..., layout=(4, 4), legend=false)
end

# ╔═╡ Cell order:
# ╟─9bbf3d20-0b30-11ec-1917-ebba9c2ee5f0
# ╟─8c102f1e-e978-4ed3-8620-830c46bb566f
# ╟─0ae615d5-739f-456f-be76-d7d3307577bb
# ╟─d67717f9-b5d6-4f84-ad80-31fe0c93b014
# ╟─a4d300fb-f712-4045-a8b8-7bf3b4b55ecc
# ╟─98d1834d-f61e-4b28-a6bc-9dec42a1bff8
# ╟─53400761-545e-4f2a-ae75-80694a43dd92
# ╠═820f64a0-287d-40fa-b5f5-2cb35dc91658
# ╟─033440c0-efc2-42c4-918a-e0d966e27d3f
# ╟─b3647ffa-d904-4bee-9327-6dec393702e5
# ╟─4f771fd7-84be-43f9-a9b2-78d40db4f91c
# ╟─6b6365a7-486a-4533-bd1e-d6c5726f3b35
# ╟─ddfc6838-cf88-4aa2-a95c-5d4c95a487ae
# ╟─66806d5b-55b7-4a4f-b592-2fe6304a7795
# ╟─de3026a5-109e-458c-9e38-8e6ab43a81c6
# ╟─5530ddd4-df5e-4308-9dfd-492f68db9f94
# ╟─c7490656-5f5f-41a5-8a56-e3341880d855
# ╟─89767b3c-63eb-4bdb-9f96-2289954908fa
# ╟─4ae5b880-1c74-4d37-b1eb-6f6f847b39a0
# ╟─6a3f7794-3776-48cf-b188-ee892078c148
# ╟─edcc9e1c-9cc0-4ead-ae6f-4c68bc509470
# ╟─18c0c325-c107-42e8-84a3-587580cc5772
# ╟─88526e35-67c0-430a-a6ef-6454cdab7d2d
# ╟─4d5c2e27-8a2c-486b-97bd-65ce4d108322
# ╟─5312759e-6c75-411b-b28f-9de6f6bfa65b
# ╟─d5fe909b-784a-443e-9e2e-c00c9e8cd611
# ╟─e3e19d9e-767a-486b-8626-108c5f70a63b
# ╟─d315c4c8-9945-409a-90b5-f08369665b25
# ╟─9658209d-57b5-4d68-a031-e15711d8cc92
# ╟─663406ed-49d2-439b-8272-69c0b29546da
# ╠═7475abbd-6fdb-423f-8981-be4461df6648
# ╟─f79ca361-a2ca-4e0e-b920-36936d01fe8b
# ╟─7a284458-83bb-48fb-863e-ff385a3ba27d
# ╟─da5d583f-b9d8-4c96-892c-c3a018d83890
# ╟─221d3ebf-e734-4a7d-8811-d8874311aa21
# ╟─58db4367-514b-4ce8-b994-a13d73af93e9
# ╟─4a0d265d-9c91-4095-899b-20e4c6765fbf
# ╟─1227c477-2fa5-4929-b64b-aeaf5745c52d
# ╟─c1b885d9-bb2b-486f-ad22-219e4ede6025
