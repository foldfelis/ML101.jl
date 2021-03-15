### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 5ebb8c30-857f-11eb-2992-296c7a001aa8
begin
	using PlutoUI
	PlutoUI.LocalResource("assets/candy.jpg")
end

# ╔═╡ ff0eaff8-857c-11eb-1945-b1541856ca36
begin 
	using DataFrames
	using CSV
	using ML101
end

# ╔═╡ cb4211fc-857b-11eb-11fe-f113a59585fc
md"
# Does it contain chocolate?

[Dataset](https://www.kaggle.com/fivethirtyeight/the-ultimate-halloween-candy-power-ranking/)

JingYu Ning
"

# ╔═╡ dd08d448-8594-11eb-2d12-bd7b6e297e37
md"
### Load Data
"

# ╔═╡ 6e409d82-857d-11eb-0d47-ed42bd67bfc4
df = CSV.read("../data/is_chocolate.csv", DataFrame)

# ╔═╡ 91932902-857e-11eb-0f43-c9cad4b77747
md"
**Field description**
* chocolate: Does it contain chocolate?
* fruity: Is it fruit flavored?
* caramel: Is there caramel in the candy?
* peanutalmondy: Does it contain peanuts, peanut butter or almonds?
* nougat: Does it contain nougat?
* crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
* hard: Is it a hard candy?
* bar: Is it a candy bar?
* pluribus: Is it one of many candies in a bag or box?
* sugarpercent: The percentile of sugar it falls under within the data set.
* pricepercent: The unit price percentile compared to the rest of the set.
* winpercent: The overall win percentage according to 269,000 matchups.
"

# ╔═╡ e94f8e72-8594-11eb-2b8f-070e354679d2
md"
### Train
"

# ╔═╡ 48af712a-8581-11eb-25ce-9bf4b270a426
begin
	features = [
		:fruity, 
		:caramel, 
		:peanutyalmondy,
		:nougat, 
		:crispedricewafer, 
		:hard,
		:bar,
		:pluribus,
		:sugarpercent,
		:pricepercent,
		:winpercent,
	]
	model = LogisticRegressionModel(df, :chocolate, features)
end;

# ╔═╡ 6fdcb82e-8582-11eb-2b4b-e3037345fde2
model.argv = [
	-4.64101,
	-0.911038,
	-0.47824,
	-7.05723,
	7.12505,
	-0.872823,
	7.15629,
	-1.21558,
	-0.328919,
	-0.305625,
	0.0486318,
];

# ╔═╡ 0bb4b842-8582-11eb-1d18-5b2f0e616d09
fit!(model, η=1.06e-2, atol=-2.3912e-1);

# ╔═╡ dd4462d4-8593-11eb-11c9-edd83fcd8b86
md"
### Prediction
"

# ╔═╡ 2c672a8e-8591-11eb-35c9-539940f0b642
function get_β(df::DataFrame, features::Vector{Symbol}, name::String)
	return Vector(df[df.candy .== name, :][1, features])
end;

# ╔═╡ e4d62fdc-8593-11eb-191b-09fb25fb3091
md"
**100 Grand**
"

# ╔═╡ dc6008ea-8592-11eb-166e-4922e69f8463
PlutoUI.LocalResource("assets/100_grand.jpg", :width => 200)

# ╔═╡ 4ecdc9de-8591-11eb-3435-93f31b8002a8
begin
	result_100_grand = predict(model, get_β(df, features, "100 Grand")) > 0.5
	md"Does 100 Grand contain chocolate? **$result_100_grand**"
end

# ╔═╡ 6a3da39e-8594-11eb-0dcd-b33d4ae809b4
md"
**Warheads**
"

# ╔═╡ 72110c82-8594-11eb-1dbc-edae553f9c44
PlutoUI.LocalResource("assets/warheads.jpg", :width => 200)

# ╔═╡ 71e0d5e0-8593-11eb-3c8e-354f59c6d95a

begin
	result_warheads = predict(model, get_β(df, features, "Warheads")) > 0.5
	md"Does Warheads contain chocolate? **$result_warheads**"
end

# ╔═╡ Cell order:
# ╟─5ebb8c30-857f-11eb-2992-296c7a001aa8
# ╟─cb4211fc-857b-11eb-11fe-f113a59585fc
# ╟─dd08d448-8594-11eb-2d12-bd7b6e297e37
# ╠═ff0eaff8-857c-11eb-1945-b1541856ca36
# ╠═6e409d82-857d-11eb-0d47-ed42bd67bfc4
# ╟─91932902-857e-11eb-0f43-c9cad4b77747
# ╟─e94f8e72-8594-11eb-2b8f-070e354679d2
# ╠═48af712a-8581-11eb-25ce-9bf4b270a426
# ╟─6fdcb82e-8582-11eb-2b4b-e3037345fde2
# ╠═0bb4b842-8582-11eb-1d18-5b2f0e616d09
# ╟─dd4462d4-8593-11eb-11c9-edd83fcd8b86
# ╠═2c672a8e-8591-11eb-35c9-539940f0b642
# ╟─e4d62fdc-8593-11eb-191b-09fb25fb3091
# ╟─dc6008ea-8592-11eb-166e-4922e69f8463
# ╠═4ecdc9de-8591-11eb-3435-93f31b8002a8
# ╟─6a3da39e-8594-11eb-0dcd-b33d4ae809b4
# ╟─72110c82-8594-11eb-1dbc-edae553f9c44
# ╠═71e0d5e0-8593-11eb-3c8e-354f59c6d95a
