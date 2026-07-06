### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 00000002-0000-0000-0000-000000000000
begin
	using Smore
	using Plots
	using Distributions
	using Random
	using Statistics
end

# ╔═╡ 00000001-0000-0000-0000-000000000000
# Launch via SmoreExamples.run_example(), or manually:
#
#   using Pluto
#   Pluto.run(notebook                   = "/path/to/replicate_runs_custom_loss.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# Custom Loss Function: Fitting to Raw Replicate Runs

SmoreBase's default `GaussianNLL` loss expects `CMData`'s pre-aggregated `μ`/`σ`. This notebook
shows how to plug in your own loss function instead — a plain Julia function wrapped in
`CustomLoss` — so `fitSurrogate` can use any aggregation strategy you choose.

As a vehicle for this example, we use raw replicate runs rather than a pre-aggregated mean/σ,
following the same custom-`AbstractCMData` pattern as `single_obs_custom_data.jl` (see that
notebook for the full walkthrough of writing a custom container/slice pair). `ReplicateCMData`
stores all `R` replicate trajectories per parameter set, and the custom loss operates on them
directly — an aggregation a plain `CMData` + `GaussianNLL` pipeline couldn't express, since it
only ever sees `μ` and `σ`.

The logistic growth SM is reused from `logistic_growth_pipeline.jl` so the two notebooks can be
compared directly.
"""

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"""
## 1  The Custom Data Type

`ReplicateCMData` stores a 5-D array of raw runs with axes
`[n_times, n_variables, n_replicates, n_conditions, n_cm_param_sets]`, and
`ReplicateCMDataSlice` is the single-parameter-set view the fitting loop sees
(`_sliceCmParamSet` drops the `n_cm_param_sets` dimension with `@view`).

The loss function is entirely separate from the data type — as a later section shows, it's just
a plain Julia function wrapped in `CustomLoss`.
"""

# ╔═╡ 00000009-0000-0000-0000-000000000000
struct ReplicateCMData <: AbstractCMData
	runs::Array{Float64,5}                 # [n_times, n_variables, n_replicates, n_conditions, n_cm_param_sets]
	times::Union{Nothing,Vector{Float64}}
end

# ╔═╡ 0000000a-0000-0000-0000-000000000000
struct ReplicateCMDataSlice <: AbstractCMDataSlice
	runs::AbstractArray{Float64,4}         # @view [n_times, n_variables, n_replicates, n_conditions]
	times::Union{Nothing,Vector{Float64}}
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
begin
	function SmoreBase._sliceCmParamSet(data::ReplicateCMData, pi::Int)
		ReplicateCMDataSlice(@view(data.runs[:, :, :, :, pi]), data.times)
	end

	SmoreBase._times(d::ReplicateCMData)      = d.times
	SmoreBase._times(d::ReplicateCMDataSlice) = d.times

	# Required by fitSurrogate for P0 validation
	SmoreBase.n_cm_param_sets(d::ReplicateCMData) = size(d.runs, 5)
end

# ╔═╡ 0000000c-0000-0000-0000-000000000000
md"""
## 2  CM Data — Stochastic Replicate Runs

We generate `R = 10` synthetic replicate trajectories. Most replicates add
small Gaussian noise (σ = 0.05) to the true logistic trajectory — introduced next as the SM —
but one replicate simulates a "runaway" event — it diverges from the logistic at late times,
motivating the MAE loss defined further below.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
# Implemented as a plain Julia function so it can double as the SM in Section 3 below.
logistic(t, p, _cond) = reshape(
	p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
	:, 1,
)

# ╔═╡ 00000007-0000-0000-0000-000000000000
begin
	t      = collect(0.0:0.5:5.0)   # 11 time points
	p_true = [0.6, 4.0]             # true r and K
end

# ╔═╡ 0000000d-0000-0000-0000-000000000000
begin
	R       = 10
	rng_gen = Random.MersenneTwister(42)
	obs_true = vec(logistic(t, p_true, nothing))

	runs = Array{Float64}(undef, length(t), 1, R, 1, 1)
	for r in 1:R
		noise = 0.05 .* randn(rng_gen, length(t))
		runs[:, 1, r, 1, 1] = obs_true .+ noise
	end

	# Replace the last replicate with a "runaway" trajectory
	runs[:, 1, R, 1, 1] = obs_true .* (1 .+ 0.5 .* t ./ maximum(t))

	data = ReplicateCMData(runs, t)
end

# ╔═╡ 0000000e-0000-0000-0000-000000000000
md"""
Shape: $(size(data.runs, 1)) times × $(size(data.runs, 3)) replicates × 1 condition × 1 param-set.
Replicate $R is the runaway outlier.
"""

# ╔═╡ 0000000f-0000-0000-0000-000000000000
let
	plt = plot(; xlabel = "Time", ylabel = "Value",
		title = "Raw replicate runs (run $R is the outlier)")
	for r in 1:(R - 1)
		plot!(plt, t, runs[:, 1, r, 1, 1]; color = :steelblue, linealpha = 0.4, linewidth = 1, label = "")
	end
	plot!(plt, t, runs[:, 1, R, 1, 1]; color = :red, linewidth = 2, label = "Outlier run $R")
	plot!(plt, t, obs_true; color = :black, linewidth = 2, linestyle = :dash, label = "True trajectory")
	plt
end

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 3  The Surrogate Model

Having observed logistic growth in the CM data above, we use a logistic growth SM — the same
function object used above to generate the CM data — same as in `logistic_growth_pipeline.jl`.
"""

# ╔═╡ 00000006-0000-0000-0000-000000000000
sm = CustomSurrogateModel(fn = logistic)

# ╔═╡ 00000010-0000-0000-0000-000000000000
md"""
## 4  Custom Loss Function

`fitSurrogate`'s loss just needs to compute a scalar from `(A_pred, slice, cm_param_set_index)`.
Here we define mean absolute error (MAE) across replicates as a plain Julia function, then wrap
it in `CustomLoss` so it slots directly into `SMFitProblem`. Any other aggregation — trimmed
mean, quantile loss, rank-based — would follow the same shape.
"""

# ╔═╡ 00000011-0000-0000-0000-000000000000
function replicate_mae(A_pred, slice::ReplicateCMDataSlice, ki::Int)
	n_rep = size(slice.runs, 3)
	total = 0.0
	for r in 1:n_rep
		total += sum(abs.(A_pred .- slice.runs[:, :, r, ki]))
	end
	return total / n_rep
end

# ╔═╡ 00000013-0000-0000-0000-000000000000
md"""
## 5  Parameter Prior
"""

# ╔═╡ 00000014-0000-0000-0000-000000000000
prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])

# ╔═╡ 00000015-0000-0000-0000-000000000000
md"""
## 6  Fitting

`SMFitProblem` accepts the wrapped loss as its `loss` keyword; `fitSurrogate` then calls it to evaluate the objective at each parameter set.
"""

# ╔═╡ 00000016-0000-0000-0000-000000000000
begin
	P0   = [0.5 5.0]
	prob = SMFitProblem(sm, data, prior; loss = CustomLoss(replicate_mae))
	result = fitSurrogate(prob, P0)
end

# ╔═╡ 00000017-0000-0000-0000-000000000000
md"""
**Fit summary**

| Parameter | True | Fitted |
|-----------|------|--------|
| r | $(p_true[1]) | $(round(result.parameters[1,1]; digits=4)) |
| K | $(p_true[2]) | $(round(result.parameters[1,2]; digits=4)) |
"""

# ╔═╡ 00000018-0000-0000-0000-000000000000
let
	t_fine = collect(range(0.0, 5.0; length = 100))
	ŷ      = vec(logistic(t_fine, result.parameters[1, :], nothing))
	ŷ_true = vec(logistic(t_fine, p_true, nothing))

	plt = plot(; xlabel = "Time", ylabel = "Value",
		title = "SM fit via custom MAE loss", legend = :topleft)
	for r in 1:(R - 1)
		plot!(plt, t, runs[:, 1, r, 1, 1];
			color = :steelblue, linealpha = 0.3, linewidth = 1, label = "")
	end
	plot!(plt, t, runs[:, 1, R, 1, 1];
		color = :red, linewidth = 2, label = "Outlier run")
	plot!(plt, t_fine, ŷ_true;
		color = :black, linewidth = 2, linestyle = :dash, label = "True")
	plot!(plt, t_fine, ŷ;
		color = :green, linewidth = 2, label = "SM fit")
	plt
end

# ╔═╡ 00000019-0000-0000-0000-000000000000
md"""
## 7  Profile Likelihood UQ

`quantifyUncertainty` calls the same custom loss during re-optimization at each profile point.
"""

# ╔═╡ 0000001a-0000-0000-0000-000000000000
uq = quantifyUncertainty(ProfileLikelihood(n_points = 25, confidence_level = 0.95), prob, result, 1)

# ╔═╡ 0000001b-0000-0000-0000-000000000000
plot(uq)

# ╔═╡ 0000001c-0000-0000-0000-000000000000
let
	rows = map(uq.profiles) do pc
		i  = pc.parameter_index
		lo = pc.ci_lower === nothing ? "—" : string(round(pc.ci_lower; digits = 4))
		hi = pc.ci_upper === nothing ? "—" : string(round(pc.ci_upper; digits = 4))
		"| $(pc.parameter_name) | $(p_true[i]) | [$lo, $hi] |"
	end
	Markdown.parse("""
**Profile likelihood confidence intervals**

| param | true | 95% CI |
|-------|------|--------|
$(join(rows, "\n"))
""")
end

# ╔═╡ 0000001d-0000-0000-0000-000000000000
md"""
## 8  Sampling SM Predictions
"""

# ╔═╡ 0000001e-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(prob, uq; nSamples = 200, rng = rng_sample)
end

# ╔═╡ 0000001f-0000-0000-0000-000000000000
plot(samples)

# ╔═╡ 00000020-0000-0000-0000-000000000000
md"""
## 9  Summary

To use a custom loss function of your own:

1. Write a plain Julia function `my_loss(A_pred, slice, cm_param_set_index) -> Float64`, using
   whatever aggregation over your data makes sense — a quantile loss, trimmed mean, rank-based
   loss, or (as here) MAE across raw replicates.
2. Wrap it: `SMFitProblem(sm, data, prior; loss = CustomLoss(my_loss))`.
3. Call `fitSurrogate(prob, P0)` as usual.

Everything downstream — profile likelihood, prediction sampling — works unchanged; `CustomLoss`
is a drop-in replacement for `GaussianNLL` everywhere in the pipeline.
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─00000008-0000-0000-0000-000000000000
# ╠═00000009-0000-0000-0000-000000000000
# ╠═0000000a-0000-0000-0000-000000000000
# ╠═0000000b-0000-0000-0000-000000000000
# ╟─0000000c-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000007-0000-0000-0000-000000000000
# ╠═0000000d-0000-0000-0000-000000000000
# ╟─0000000e-0000-0000-0000-000000000000
# ╠═0000000f-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╟─00000010-0000-0000-0000-000000000000
# ╠═00000011-0000-0000-0000-000000000000
# ╟─00000013-0000-0000-0000-000000000000
# ╠═00000014-0000-0000-0000-000000000000
# ╟─00000015-0000-0000-0000-000000000000
# ╠═00000016-0000-0000-0000-000000000000
# ╟─00000017-0000-0000-0000-000000000000
# ╠═00000018-0000-0000-0000-000000000000
# ╟─00000019-0000-0000-0000-000000000000
# ╠═0000001a-0000-0000-0000-000000000000
# ╠═0000001b-0000-0000-0000-000000000000
# ╠═0000001c-0000-0000-0000-000000000000
# ╟─0000001d-0000-0000-0000-000000000000
# ╠═0000001e-0000-0000-0000-000000000000
# ╠═0000001f-0000-0000-0000-000000000000
# ╟─00000020-0000-0000-0000-000000000000
