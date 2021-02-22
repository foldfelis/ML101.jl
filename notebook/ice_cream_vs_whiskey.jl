### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ abde3380-750f-11eb-119b-5dc493c19b15
begin
	using CSV
	using DataFrames 
	using Dates
	using Plots
	using StatsPlots
	plotly()
end

# ╔═╡ da1eaa04-750f-11eb-185c-c11c2c913513
raw = CSV.read("../data/ice_cream_vs_whiskey_trends.csv", DataFrame)

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

# ╔═╡ e932d9fa-7513-11eb-1a99-5963f9568df3
begin
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
	
	raw.TempIndex = map(gen_temp_index, raw.Date)
	raw.IsTempInc = map(gen_is_temp_inc, raw.Date)
	raw
end

# ╔═╡ 77099666-751b-11eb-206d-636cdff374f2
begin
	@df raw scatter(:TempIndex, :IceCreamT, label="Ice Cream", group=:IsTempInc)
	@df raw scatter!(:TempIndex, :WhiskeyT, label="Whiskey", group=:IsTempInc)
	plot!(
		title="Trends",
		xlabel="Temperature Index (arb. unit)",
		ylabel="Percentage",
		# xlims=(raw.Date[1]-Month(2), raw.Date[end]+Month(2)),
		# xticks=collect(raw.Date[1]:Month(6):raw.Date[end]+Week(1)),
		size=(650, 400)
	)
end

# ╔═╡ Cell order:
# ╠═abde3380-750f-11eb-119b-5dc493c19b15
# ╠═da1eaa04-750f-11eb-185c-c11c2c913513
# ╠═31120810-7510-11eb-36ec-db5fc38cbf89
# ╠═e932d9fa-7513-11eb-1a99-5963f9568df3
# ╠═77099666-751b-11eb-206d-636cdff374f2
