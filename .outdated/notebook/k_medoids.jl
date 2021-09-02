### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 7c28cf4c-cd0c-11eb-0507-b10f2a883e8c
begin
    using RDatasets
    using LinearAlgebra
    using Plots
    using StatsPlots
    plotly(size=(680, 500))
end

# ╔═╡ 93477f7b-aef4-44e5-ba3b-8285a6261500
md"
# K-Medoids

JingYu Ning
"

# ╔═╡ 64f0d05d-9c4e-4935-bf33-a5ee5835f99a
md"
## IRIS dataset
"

# ╔═╡ 7f59bdb6-c6de-4a25-9ff0-8a81457a4242
iris = dataset("datasets", "iris")

# ╔═╡ 63073482-7add-4234-ab34-6c8068b2f5a6
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

# ╔═╡ fde3d98a-be02-47fe-a8c1-293092298659
plot(gen_pots()..., layout=(4, 4), legend=false)

# ╔═╡ 30f8c663-e43b-44e2-9c91-b3e8dd760e5f
md"
## Initial center points
"

# ╔═╡ b1b771a2-3e94-4666-a668-01dee57f9832
md"
**Data size and numbers of classes**
"

# ╔═╡ 52e9ada2-cff4-4fe4-abb0-ec0ebc2d3fa1
begin
    n = size(iris, 1)
    n_classes = 3
end;

# ╔═╡ cbb9a475-36d9-4c9f-a12a-d2cc5c339be7
md"
**Initial three center points**
"

# ╔═╡ ad7bfec2-3fe8-428c-ac92-ad56183cbcdd
# cs_i = rand(1:n, n_classes)
cs_i = [77, 90, 13]

# ╔═╡ 2c6511ca-e070-47ac-8f2e-c1308f491f98
let
    plots = gen_pots()
    for i in 2:4, j in 1:(i-1)
        plots[i, j] = @df iris[cs_i, :] scatter!(plots[i, j], cols(i), cols(j))
    end
    plot(plots..., layout=(4, 4), legend=false)
end

# ╔═╡ bb476b47-ba34-41d7-a32f-fc204f4a328c
md"
## Training
"

# ╔═╡ f558898f-d426-4a36-baed-fa4dd07cf7b2
let
	data = randn(2, 1000)
	data_o = hcat(data, [9, -9])
	
	function d²(data::Matrix, i::Integer, j::Integer)
		return sum(abs.(data[:, j]-data[:, i]))
	end
	
	xlim=(-10, 10); ylim=(-10, 10)
	
	pd = scatter(data[1, :], data[2, :], xlim=xlim, ylim=ylim)
	id = argmin([sum(d²(data, i, j) for j in 1:1000) for i in 1:1000])
	scatter!(pd, [data[1, id]], [data[2, id]])
	
	po = scatter(data_o[1, :], data_o[2, :], xlim=xlim, ylim=ylim)
	io = argmin([sum(d²(data_o, i, j) for j in 1:1000) for i in 1:1000])
	scatter!(po, [data_o[1, io]], [data_o[2, io]])
	plot(pd, po, legend=false)
end

# ╔═╡ e6f38859-fd04-4dd6-84f3-991fdc72317d
md"
**Manhattan distance between two points**
"

# ╔═╡ 9ce95067-b54f-4a9d-8158-8e25b4a2480f
function d(data::DataFrame, p₁_i::Integer, p₂_i::Integer)
	return sum(abs.(Vector(data[p₁_i, 1:4]) - Vector(data[p₂_i, 1:4])))
end

# ╔═╡ 9cf597b7-9ea2-4c5d-b810-83b5d027d758
begin
    signal = "⚡"

    iris[!, :Predict] = zeros(n)
    for _ in 1:100
        iris[!, :Predict] = map(i->argmin([d(iris, i, ci) for ci in cs_i]), 1:n)

		iris[!, :SumInClassD] = [
			sum((iris[i, :Predict]==iris[j, :Predict] && d(iris, i, j)) for j in 1:n)
			for i in 1:n
		]
		for i in 1:n
			if iris[cs_i[iris[i, :Predict]], :SumInClassD] > iris[i, :SumInClassD]
				cs_i[iris[i, :Predict]] = i
			end
		end
    end
end

# ╔═╡ 7b5ea310-f5c2-4f9d-91b0-2ebae7b8639e
md"
**Prediction**
"

# ╔═╡ e0b73f20-18ba-4dff-9b75-0fe82d072167
let
    signal
    plots = gen_pots()
    for i in 2:4, j in 1:(i-1)
        plots[i, j] = @df iris scatter!(plots[i, j], cols(i), cols(j), group=:Predict)
        plots[i, j] = @df iris[cs_i, :] scatter!(plots[i, j], cols(i), cols(j))
    end
    plot(plots..., layout=(4, 4), legend=false)
end

# ╔═╡ 413aa844-5ce1-4349-a715-b62c8db9a1f8
md"
**Ground truth**
"

# ╔═╡ 1ae9328e-72c1-46e4-ba1e-742205c4bd88
let
    plots = gen_pots()
    for i in 2:4, j in 1:(i-1)
        plots[i, j] = @df iris scatter!(plots[i, j], cols(i), cols(j), group=:Species)
    end
    plot(plots..., layout=(4, 4), legend=false)
end

# ╔═╡ Cell order:
# ╟─93477f7b-aef4-44e5-ba3b-8285a6261500
# ╠═7c28cf4c-cd0c-11eb-0507-b10f2a883e8c
# ╟─64f0d05d-9c4e-4935-bf33-a5ee5835f99a
# ╠═7f59bdb6-c6de-4a25-9ff0-8a81457a4242
# ╟─63073482-7add-4234-ab34-6c8068b2f5a6
# ╟─fde3d98a-be02-47fe-a8c1-293092298659
# ╟─30f8c663-e43b-44e2-9c91-b3e8dd760e5f
# ╟─b1b771a2-3e94-4666-a668-01dee57f9832
# ╠═52e9ada2-cff4-4fe4-abb0-ec0ebc2d3fa1
# ╟─cbb9a475-36d9-4c9f-a12a-d2cc5c339be7
# ╠═ad7bfec2-3fe8-428c-ac92-ad56183cbcdd
# ╟─2c6511ca-e070-47ac-8f2e-c1308f491f98
# ╟─bb476b47-ba34-41d7-a32f-fc204f4a328c
# ╟─f558898f-d426-4a36-baed-fa4dd07cf7b2
# ╟─e6f38859-fd04-4dd6-84f3-991fdc72317d
# ╠═9ce95067-b54f-4a9d-8158-8e25b4a2480f
# ╠═9cf597b7-9ea2-4c5d-b810-83b5d027d758
# ╟─7b5ea310-f5c2-4f9d-91b0-2ebae7b8639e
# ╟─e0b73f20-18ba-4dff-9b75-0fe82d072167
# ╟─413aa844-5ce1-4349-a715-b62c8db9a1f8
# ╟─1ae9328e-72c1-46e4-ba1e-742205c4bd88
