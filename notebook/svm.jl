### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 3ec6aae0-c233-11eb-355d-3dfe780e33bc
begin
	using DataFrames
	using JuMP
	using Ipopt
	using StatsPlots
	plotly()
end

# ╔═╡ a6997248-aee3-4ebb-8a1e-f791ad503aa8
begin
	n = 100
	μ1 = 10
	μ2 = 15
	
	df = append!(
		DataFrame(x1=randn(n).+μ1, x2=randn(n).+μ1, y=ones(n)),
		DataFrame(x1=randn(n).+μ2, x2=randn(n).+μ2, y=-ones(n))
	)
	
	p = @df df scatter(:x1, :x2, group=:y, legend=:topleft)
end

# ╔═╡ 5056cc0a-f832-47b3-a6a5-f7ee2d7bc5ba
begin
	x = Matrix(df[:, 1:2])
	y = Vector(df[:, 3])
	
	function solve_svm(x, y)
		svm = Model(Ipopt.Optimizer)
		@variable(svm, w[1:2])
		@variable(svm, b)
		@objective(svm, Min, 0.5 * w' * w)
		@constraint(svm, y.*(x * w .+ b) .>= 1)
		optimize!(svm)

		@show objective_value(svm)
		
		return value.(w), value(b)
	end
	
	w, b = solve_svm(x, y)
end

# ╔═╡ e5334dab-56b2-4dd0-a12c-e17f1c8e584b
ŷ(x1, x2) = sign([x1, x2]' * w + b)

# ╔═╡ 16448a5a-090f-4a6a-a9a0-55192a77dfb3
heatmap!(p, 5:0.1:20, 5:0.1:20, ŷ, color=:coolwarm)

# ╔═╡ bf7d0459-4b47-4839-800d-01bdcd500633
md"
$arg\ min_{𝐰, b}\ \frac{1}{2}𝐰^T𝐰$

$\text{subject to }\forall i, y_i(𝐰^T𝐱_i+b) \geq 1$
"

# ╔═╡ 27dfce4e-3ccd-4221-8eca-144ae87e2998
md"
respect to 

$arg\ min_{𝐰, b}\ \frac{1}{2}𝐮^T 𝐐 𝐮 + 𝐏^T 𝐮$

$\text{subject to }\forall i, 𝐚_i^T 𝐮) \geq c_i$

Therefore, 

$𝐮 = \begin{bmatrix}b \\ 𝐰\end{bmatrix}$
$𝐐 = \begin{bmatrix}0 & 0^T_d \\ 0^T_d & 𝐈_d\end{bmatrix}$
$𝐏 = 0_{d+1}$
$𝐚^T_i = y_i\begin{bmatrix}1 & 𝐱^T_i\end{bmatrix}$
$c_i = 1$
"

# ╔═╡ Cell order:
# ╠═3ec6aae0-c233-11eb-355d-3dfe780e33bc
# ╠═a6997248-aee3-4ebb-8a1e-f791ad503aa8
# ╟─bf7d0459-4b47-4839-800d-01bdcd500633
# ╟─27dfce4e-3ccd-4221-8eca-144ae87e2998
# ╠═5056cc0a-f832-47b3-a6a5-f7ee2d7bc5ba
# ╠═e5334dab-56b2-4dd0-a12c-e17f1c8e584b
# ╠═16448a5a-090f-4a6a-a9a0-55192a77dfb3
