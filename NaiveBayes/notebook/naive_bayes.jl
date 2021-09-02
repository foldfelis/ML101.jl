### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 21c74f60-fe61-4b5f-8fe5-1840b0a4bcb0
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 705f3bc2-ea0a-11eb-2d81-21de5db26c02
begin
	using CSV
	using DataFrames
	using WordCloud
end

# ╔═╡ 30b58656-7d24-4ff3-908f-c372de38fc9b
md"
## Preprocess
"

# ╔═╡ c9d86708-8ce9-48dd-8d18-ba04b4c4a22d
sms2token(sms::String) = [m.match for m in eachmatch(r"[a-z']+", lowercase(sms))]

# ╔═╡ a7b1da07-fa68-44c9-b01d-b6b0196f02b1
begin
	df_raw = CSV.read("../data/SMSSpamCollection", DataFrame, header=[:Label, :SMS])
	
	df = copy(df_raw) 
	df[!, :SMS] .= sms2token.(df.SMS)
	
	spam_df = df[df.Label.=="spam", :]
	ham_df = df[df.Label.=="ham", :]
	
	# df_raw
	df
end

# ╔═╡ 6d356343-6f13-4792-b084-b55aa0af7510
md"
## Text mining
"

# ╔═╡ b4a30d03-5f3c-4fb3-9351-9c8f689032a2
begin
	features = unique(vcat(df.SMS...))
	
	function calc_tf(words::Vector)
		tf = Dict([(w, Float64(count(x->x==w, words))) for w in features])
		
		n = sum(values(tf))
		for (k, v) in tf
			tf[k] = (v == 0) ? 1e-1/n : v/n
		end
		
		return tf
	end
end

# ╔═╡ f10b2755-70d1-4cfe-a759-17f283bb7cb3
begin
	spam_p = nrow(spam_df) / nrow(df)
	spam_tf = calc_tf(vcat(spam_df.SMS...))
	
	ham_p = nrow(ham_df) / nrow(df)
	ham_tf = calc_tf(vcat(ham_df.SMS...))
end;

# ╔═╡ ac757e08-0b19-4aa8-a791-8a677280b12f
function plot_word_cloud(tf::Dict, top_n::Integer)
	tf = sort(collect(tf), by=x->x[2])
	
	return wordcloud(
		map(x->x[1], tf)[1:top_n], 
		map(x->x[2], tf)[1:top_n],
		font="LiSong Pro",
		mask=shape(box, 400*2, 300*2, 10*2),
		colors=:Dark2_3,
		angles=(0, 90),
		density=0.4
	)
end

# ╔═╡ 7f30886f-32e3-498c-8166-ea5f18b5e725
md"
### Non-spam word cloud
"

# ╔═╡ d9127fa7-e83d-4c21-a2d0-9fa497e1af44
plot_word_cloud(spam_tf, 400)

# ╔═╡ 1fca2c23-5ff9-42cc-a5fa-a872d4cf0cab
md"
### Spam word cloud
"

# ╔═╡ b4d6a0ed-8e41-44d1-b736-e3acc1cd511e
plot_word_cloud(ham_tf, 400)

# ╔═╡ e7997329-45fa-42cb-8d6b-399089099110
md"
## Model
"

# ╔═╡ dcb7da4c-5092-4667-845a-af4d3cd7c9d3
md"
$P(Cs|\vec{x}) = \frac{P(\vec{x}|Cs) P(Cs)}{P(\vec{x})}$

$P(Cs|\vec{x}) = \frac{P(\vec{x}|Cs) P(Cs)}{P(\vec{x}|Cs)P(Cs) + P(\vec{x}|Ch)P(Ch)}$

$P(\vec{x}|C) = P(w1|C)P(w2|C)P(w3|C)...$
"

# ╔═╡ a3f53b05-f2dd-46af-b573-3c5de2c3f709
begin
	function p_x_c(tokens::Vector, tf::Dict)
		p = 1
		for t in tokens
			p *= tf[t]
		end
		
		return p
	end
	
	function infer(s::String)
		tokens = filter(x->x in features, sms2token(s))
		(isempty(tokens)) && (return 0.5)

		return p_x_c(tokens, spam_tf)*spam_p / (
			p_x_c(tokens, spam_tf)*spam_p + p_x_c(tokens, ham_tf)*ham_p
		)
	end
end

# ╔═╡ f95f4e29-44ec-48ae-8ef3-878c3f88ba4c
md"
## Validation

Is the following message a spam?
"

# ╔═╡ 60b39235-a767-452e-998a-6f6efa0a7c1e
infer(
	"Congratulations! You have won a free launch today~~~"
)

# ╔═╡ 393e099f-135c-4861-8581-fb5a48066df4
infer(
	"I am the most handsome guy around the world!!"
)

# ╔═╡ Cell order:
# ╟─21c74f60-fe61-4b5f-8fe5-1840b0a4bcb0
# ╠═705f3bc2-ea0a-11eb-2d81-21de5db26c02
# ╟─30b58656-7d24-4ff3-908f-c372de38fc9b
# ╠═c9d86708-8ce9-48dd-8d18-ba04b4c4a22d
# ╠═a7b1da07-fa68-44c9-b01d-b6b0196f02b1
# ╟─6d356343-6f13-4792-b084-b55aa0af7510
# ╠═b4a30d03-5f3c-4fb3-9351-9c8f689032a2
# ╠═f10b2755-70d1-4cfe-a759-17f283bb7cb3
# ╠═ac757e08-0b19-4aa8-a791-8a677280b12f
# ╟─7f30886f-32e3-498c-8166-ea5f18b5e725
# ╠═d9127fa7-e83d-4c21-a2d0-9fa497e1af44
# ╟─1fca2c23-5ff9-42cc-a5fa-a872d4cf0cab
# ╠═b4d6a0ed-8e41-44d1-b736-e3acc1cd511e
# ╟─e7997329-45fa-42cb-8d6b-399089099110
# ╟─dcb7da4c-5092-4667-845a-af4d3cd7c9d3
# ╠═a3f53b05-f2dd-46af-b573-3c5de2c3f709
# ╟─f95f4e29-44ec-48ae-8ef3-878c3f88ba4c
# ╠═60b39235-a767-452e-998a-6f6efa0a7c1e
# ╠═393e099f-135c-4861-8581-fb5a48066df4
