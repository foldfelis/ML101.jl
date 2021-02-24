### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 51259b56-7567-11eb-16e3-63b248b060cd
begin
	using PlutoUI
	LocalResource("./assets/ice_cream_vs_whiskey.png")
end

# ╔═╡ abde3380-750f-11eb-119b-5dc493c19b15
begin
	using ML101
	using CSV
	using DataFrames
	using Dates
	using Plots
	using StatsPlots
	plotly()
end;

# ╔═╡ 62b389c8-755a-11eb-3c5e-0f650123135c
md"
# Linear Regression Analysis

#### Ice cream vs Whiskey

JingYu Ning
"

# ╔═╡ 9d2bad86-755a-11eb-1ac8-297ba703a65d
md"
## Load data
"

# ╔═╡ da1eaa04-750f-11eb-185c-c11c2c913513
begin
	raw = CSV.read("../data/ice_cream_vs_whiskey_trends.csv", DataFrame)
	raw
end

# ╔═╡ c67bb8fc-755a-11eb-3338-a9cea1251f3c
md"
## Raw data visualization
"

# ╔═╡ 31120810-7510-11eb-36ec-db5fc38cbf89
begin
	@df raw plot(:Date, :IceCreamT, label="Ice Cream", lw=2)
	@df raw plot!(:Date, :WhiskeyT, label="Whiskey", lw=2)
	plot!(
		title="Trends",
		xlabel="Date",
		ylabel="Percentage",
		xlims=(raw.Date[1]-Month(2), raw.Date[end]+Month(2)),
		xticks=collect(raw.Date[1]:Month(6):raw.Date[end]+Week(1)),
		size=(650, 400)
	)
end

# ╔═╡ 10bd9ec6-755b-11eb-19f8-b71750df45b2
md"
## Preprocess
"

# ╔═╡ e932d9fa-7513-11eb-1a99-5963f9568df3
begin
	df = copy(raw)
	function gen_temp_index(date::Date)
		if Month(date) == Month(1)
			temp_index = abs(date - Date(Year(date).value, 1, 15))
		else
			if dayofyear(date) <= dayofyear(Date(Year(date).value, 7, 15))
				temp_index = date - Date(Year(date).value, 1, 15)
			else
				temp_index = Date(Year(date).value+1, 1, 15) - date
			end
		end

		return temp_index.value
	end

	function gen_is_temp_inc(date::Date)
		Date(Year(date).value,1,15)<date<=Date(Year(date).value,7,15) ? true : false
	end

	df.TempIndex = map(gen_temp_index, df.Date) ./ 182.5
	df.IsTempInc = map(gen_is_temp_inc, df.Date)
	df[!, :IceCreamT] = df.IceCreamT ./ 100
	df[!, :WhiskeyT] = df.WhiskeyT ./ 100
	df
end

# ╔═╡ 8f52c950-755b-11eb-0256-9767e13d82f4
md"
## Is trends Temperature dependent?
"

# ╔═╡ a6adc136-755b-11eb-1a70-bd9f483b1bcf
begin
	@df df scatter(:TempIndex, :IceCreamT, label="Ice Cream")
	@df df scatter!(:TempIndex, :WhiskeyT, label="Whiskey")
	plot!(
		title="Trends",
		xlabel="Temperature Index (arb. unit)",
		ylabel="Percentage",
		xlims=(-5e-2, 1+5e-2),
		size=(650, 400)
	)
end

# ╔═╡ 7a4cc232-7564-11eb-3cfb-b727d76a46f5
begin
	lrm_ice_cream_temp = LinearRegressionModel(
		df, label=:IceCreamT, features=[:TempIndex]
	)
	lrm_ice_cream_temp.argv = [0.4, 0.4]
	fit!(lrm_ice_cream_temp, lr=1e-7, atol=8e-3, show=true)

	lrm_whiskey_temp = LinearRegressionModel(
		df, label=:WhiskeyT, features=[:TempIndex]
	)
	lrm_whiskey_temp.argv = [0.45, -0.1]
	fit!(lrm_whiskey_temp, lr=1e-7, atol=6e-3, show=true)
end

# ╔═╡ a158c064-7560-11eb-3a9c-c77cab156802
begin
	@df df scatter(:TempIndex, :IceCreamT, label="Ice Cream")
	@df df scatter!(:TempIndex, :WhiskeyT, label="Whiskey")
	plot!(
		df.TempIndex,
		x->predict(lrm_ice_cream_temp, [x]),
		label="Ice Cream Model",
		lw=5,
		color=ARGB(0, 0.5, 1)
	)
	plot!(
		df.TempIndex,
		x->predict(lrm_whiskey_temp, [x]),
		label="Whiskey Model",
		lw=5,
		color=ARGB(1, 0.5, 0)
	)
	plot!(
		title="Trends",
		xlabel="Temperature Index (arb. unit)",
		ylabel="Percentage",
		xlims=(-5e-2, 1+5e-2),
		size=(650, 400),
		legend=:topleft
	)
end

# ╔═╡ Cell order:
# ╟─62b389c8-755a-11eb-3c5e-0f650123135c
# ╟─51259b56-7567-11eb-16e3-63b248b060cd
# ╟─abde3380-750f-11eb-119b-5dc493c19b15
# ╟─9d2bad86-755a-11eb-1ac8-297ba703a65d
# ╟─da1eaa04-750f-11eb-185c-c11c2c913513
# ╟─c67bb8fc-755a-11eb-3338-a9cea1251f3c
# ╟─31120810-7510-11eb-36ec-db5fc38cbf89
# ╟─10bd9ec6-755b-11eb-19f8-b71750df45b2
# ╟─e932d9fa-7513-11eb-1a99-5963f9568df3
# ╟─8f52c950-755b-11eb-0256-9767e13d82f4
# ╟─a6adc136-755b-11eb-1a70-bd9f483b1bcf
# ╠═7a4cc232-7564-11eb-3cfb-b727d76a46f5
# ╟─a158c064-7560-11eb-3a9c-c77cab156802
