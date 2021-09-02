### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 1d36ffb4-47cf-479c-add4-c56ef1b1a860
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 200b3ca5-0648-489d-b10d-62a851cfdb4a
begin
	using Plots
	plotly()
end

# ╔═╡ 2042164a-c8f4-48ba-82e7-330205b1d942
md"
# 找一個點，取代一個群
"

# ╔═╡ aecba716-d27b-11eb-10e0-f529b2367d05
data = vcat(randn(400, 2).-0.5, 2randn(201, 2).+1);

# ╔═╡ 51d14132-f210-4982-87b0-9626bd185ae4
begin
	scatter(data[:, 1], data[:, 2], label="data")

	mean = [sum(data[:, 1])/601, sum(data[:, 2])/601]
	scatter!([mean[1]], [mean[2]], label="mean", markersize=10)

	medoid_i = argmin(
		[sum(sum(abs.(data[j, :]-data[i, :])) for j in 1:601) for i in 601]
	)
	scatter!([data[1, medoid_i]], [data[2, medoid_i]], label="medoid", markersize=10)

	median = [sort(data[:, 1])[301], sort(data[:, 2])[301]]
	scatter!([median[1]], [median[2]], label="median", markersize=10)

	mode = [-0.5, -0.5]
	scatter!([mode[1]], [mode[2]], label="mode", markersize=10)
end

# ╔═╡ fc7216ea-86c2-4822-8ceb-a2f9d332dd48
md"
## 中心 (medoid)：

定義 'swapping cust' = $J = argmin(\sum_{j=1}^m \sum_{j=1}^n d(x_i, c_j))$

中心所在之位置滿足與其他點之距離和最小

即在座標空間中，所有點繞著他轉
"

# ╔═╡ 28fdfd71-6505-4c48-b6df-506eb6385572
md"
## 中位 (median)：

中位數落在資料空間之正中間
"

# ╔═╡ a1ef3f18-d27a-416d-97af-a7efa5299d94
md"
## 眾數 (mode)：

眾數落在座標空間中資料密度最大的地方
"

# ╔═╡ bd8d7cae-107a-4181-a2d5-5ad3815b4995
md"---"

# ╔═╡ d5e47835-0f8a-4dbd-9ac0-904868a0d818
md"
## K-Medoid

1. 在有限步數內收斂
2. O(n²)
3. 初始值不能亂給
4. 因初始值的不確定性而不保證收斂到全域最小值
5. 對於類別型變數無法定義 1-norm
"

# ╔═╡ c0417983-bb49-4797-aa62-29be1db7d4d8
md"
## K-Mode

1. 可處理類別型變數 (K-Means 也不行)
2. 使用 cosine similarity 來構造距離
"

# ╔═╡ f7dd3999-9e32-498a-bd3f-35c2526e452f
md"
## K-Prototypes

1. 可處理類別型變數以及數執行變數
2. 使用歐式距離以及 cosine similarity 的線性組合來構造距離
"

# ╔═╡ Cell order:
# ╟─2042164a-c8f4-48ba-82e7-330205b1d942
# ╟─1d36ffb4-47cf-479c-add4-c56ef1b1a860
# ╠═200b3ca5-0648-489d-b10d-62a851cfdb4a
# ╠═aecba716-d27b-11eb-10e0-f529b2367d05
# ╟─51d14132-f210-4982-87b0-9626bd185ae4
# ╟─fc7216ea-86c2-4822-8ceb-a2f9d332dd48
# ╟─28fdfd71-6505-4c48-b6df-506eb6385572
# ╟─a1ef3f18-d27a-416d-97af-a7efa5299d94
# ╟─bd8d7cae-107a-4181-a2d5-5ad3815b4995
# ╟─d5e47835-0f8a-4dbd-9ac0-904868a0d818
# ╟─c0417983-bb49-4797-aa62-29be1db7d4d8
# ╟─f7dd3999-9e32-498a-bd3f-35c2526e452f
