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
# Custom CM Data: Raw Replicate Runs with a Custom Loss

`CMData` stores pre-aggregated summary statistics: a mean array `μ` and a
standard deviation array `σ`. The default `GaussianNLL` loss then treats the
CM as having produced Gaussian observations centred at `μ` with spread `σ`.

This aggregation is convenient, but it commits you to a specific noise model
before fitting begins and discards the raw replicate trajectories. Two
situations where you may want to keep the raw runs:

- **Non-Gaussian distributions.** ABMs with extinction or runaway events
  produce bimodal run ensembles; aggregating to `μ ± σ` misrepresents the
  distribution.
- **Robust fitting.** Mean squared error (MSE) is sensitive to outlier runs;
  mean absolute error (MAE) is not. To use MAE you need access to individual
  runs — you cannot reconstruct it from `μ` and `σ` alone.

This notebook shows how to define `ReplicateCMData`, which stores all `R`
replicate trajectories, and a custom MAE loss that fits the SM to the **median**
run behaviour rather than the mean. A bonus comparison cell shows how to swap
to MSE (equivalent to fitting the mean) and how the two loss functions produce
different confidence intervals when outlier runs are present.

The logistic growth SM is reused from `logistic_growth_pipeline.jl` so the
two notebooks can be compared directly.
"""

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 1  The Surrogate Model

Same logistic growth SM as in `logistic_growth_pipeline.jl`.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
logistic(t, p, _cond) = reshape(
	p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
	:, 1,
)

# ╔═╡ 00000006-0000-0000-0000-000000000000
sm = AnalyticalSurrogateModel(fn = logistic)

# ╔═╡ 00000007-0000-0000-0000-000000000000
begin
	t      = collect(0.0:0.5:5.0)   # 11 time points
	p_true = [0.6, 4.0]             # true r and K
end

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"""
## 2  The Custom Data Type

Three pieces are needed:

1. **`ReplicateCMData`** — stores a 5-D array of raw runs with axes
   `[n_times, n_variables, n_replicates, n_conditions, n_param_sets]`.
   Replicates (dim 3) sit adjacent to the per-run data; conditions (dim 4)
   and param sets (dim 5) are the outer experimental axes.

2. **`ReplicateCMDataSlice`** — a single-parameter-set view of the runs.

3. **`_sliceParamSet`** — drops the `n_param_sets` dimension with `@view`,
   returning a `ReplicateCMDataSlice`.

The loss function is separate from the type — it's a plain Julia function
wrapped in `CustomLoss`, so you can swap aggregation strategies (MAE → MSE,
or median → trimmed mean) without changing the data type.
"""

# ╔═╡ 00000009-0000-0000-0000-000000000000
struct ReplicateCMData <: AbstractCMData
	runs::Array{Float64,5}                 # [n_times, n_variables, n_replicates, n_conditions, n_param_sets]
	times::Union{Nothing,Vector{Float64}}
end

# ╔═╡ 0000000a-0000-0000-0000-000000000000
struct ReplicateCMDataSlice <: AbstractCMDataSlice
	runs::AbstractArray{Float64,4}         # @view [n_times, n_variables, n_replicates, n_conditions]
	times::Union{Nothing,Vector{Float64}}
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
begin
	function SmoreBase._sliceParamSet(data::ReplicateCMData, pi::Int)
		ReplicateCMDataSlice(@view(data.runs[:, :, :, :, pi]), data.times)
	end

	SmoreBase._times(d::ReplicateCMData)      = d.times
	SmoreBase._times(d::ReplicateCMDataSlice) = d.times

	# Required by fitSurrogate for P0 validation
	SmoreBase.n_param_sets(d::ReplicateCMData) = size(d.runs, 5)
end

# ╔═╡ 0000000c-0000-0000-0000-000000000000
md"""
## 3  CM Data — Stochastic Replicate Runs

We generate `R = 10` synthetic replicate trajectories. Most replicates add
small Gaussian noise (σ = 0.05) to the true logistic trajectory, but one
replicate simulates a "runaway" event — it diverges from the logistic at late
times. This outlier will affect MSE fitting but not MAE fitting.
"""

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

# ╔═╡ 00000010-0000-0000-0000-000000000000
md"""
## 4  Loss Functions

We define two loss functions and wrap each in `CustomLoss`:

- **MAE** — mean absolute error across replicates, equivalent to minimising
  residuals against the **median** run. Robust to the outlier.
- **MSE** — mean squared error across replicates, equivalent to minimising
  residuals against the **mean** run. Pulled toward the outlier.

Defining them as plain functions (rather than anonymous closures) makes them
independently testable and easy to read.
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

# ╔═╡ 00000012-0000-0000-0000-000000000000
function replicate_mse(A_pred, slice::ReplicateCMDataSlice, ki::Int)
	n_rep = size(slice.runs, 3)
	total = 0.0
	for r in 1:n_rep
		diff = A_pred .- slice.runs[:, :, r, ki]
		total += sum(diff .^ 2)
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
## 6  Fitting — MAE vs MSE

`SMFitProblem` accepts a `loss` keyword, so we create one problem per loss
strategy and fit both to compare the results.
"""

# ╔═╡ 00000016-0000-0000-0000-000000000000
begin
	P0       = [0.5 5.0]
	prob_mae = SMFitProblem(sm, data, prior; loss = CustomLoss(replicate_mae))
	prob_mse = SMFitProblem(sm, data, prior; loss = CustomLoss(replicate_mse))
	result_mae = fitSurrogate(prob_mae, P0)
	result_mse = fitSurrogate(prob_mse, P0)
end

# ╔═╡ 00000017-0000-0000-0000-000000000000
md"""
**Fit comparison**

| Parameter | True | MAE fit | MSE fit |
|-----------|------|---------|---------|
| r | $(p_true[1]) | $(round(result_mae.parameters[1,1]; digits=4)) | $(round(result_mse.parameters[1,1]; digits=4)) |
| K | $(p_true[2]) | $(round(result_mae.parameters[1,2]; digits=4)) | $(round(result_mse.parameters[1,2]; digits=4)) |

The MSE fit is pulled toward the outlier run (higher K), while the MAE fit
stays close to the true parameters.
"""

# ╔═╡ 00000018-0000-0000-0000-000000000000
let
	t_fine    = collect(range(0.0, 5.0; length = 100))
	ŷ_mae     = vec(logistic(t_fine, result_mae.parameters[1, :], nothing))
	ŷ_mse     = vec(logistic(t_fine, result_mse.parameters[1, :], nothing))
	ŷ_true    = vec(logistic(t_fine, p_true, nothing))

	plt = plot(; xlabel = "Time", ylabel = "Value",
		title = "MAE vs MSE fit (one outlier replicate)", legend = :topleft)
	for r in 1:(R - 1)
		plot!(plt, t, runs[:, 1, r, 1, 1];
			color = :steelblue, linealpha = 0.3, linewidth = 1, label = "")
	end
	plot!(plt, t, runs[:, 1, R, 1, 1];
		color = :red, linewidth = 2, label = "Outlier run")
	plot!(plt, t_fine, ŷ_true;
		color = :black, linewidth = 2, linestyle = :dash, label = "True")
	plot!(plt, t_fine, ŷ_mae;
		color = :green, linewidth = 2, label = "MAE fit (robust)")
	plot!(plt, t_fine, ŷ_mse;
		color = :orange, linewidth = 2, label = "MSE fit (pulled by outlier)")
	plt
end

# ╔═╡ 00000019-0000-0000-0000-000000000000
md"""
## 7  Profile Likelihood UQ

Profile likelihood UQ runs identically for both fits. We compute both so we
can compare confidence intervals.
"""

# ╔═╡ 0000001a-0000-0000-0000-000000000000
begin
	uq_mae = quantifyUncertainty(prob_mae, result_mae, ProfileLikelihood(n_points = 25, confidence_level = 0.95))
	uq_mse = quantifyUncertainty(prob_mse, result_mse, ProfileLikelihood(n_points = 25, confidence_level = 0.95))
end

# ╔═╡ 0000001b-0000-0000-0000-000000000000
plot(uq_mae)

# ╔═╡ 0000001c-0000-0000-0000-000000000000
let
	rows = map(uq_mae.profiles) do pc
		i   = pc.parameter_index
		lo_mae = pc.ci_lower === nothing ? "—" : string(round(pc.ci_lower; digits = 4))
		hi_mae = pc.ci_upper === nothing ? "—" : string(round(pc.ci_upper; digits = 4))
		pc_mse = uq_mse.profiles[i]
		lo_mse = pc_mse.ci_lower === nothing ? "—" : string(round(pc_mse.ci_lower; digits = 4))
		hi_mse = pc_mse.ci_upper === nothing ? "—" : string(round(pc_mse.ci_upper; digits = 4))
		"| $(pc.parameter_name) | $(p_true[i]) | [$lo_mae, $hi_mae] | [$lo_mse, $hi_mse] |"
	end
	Markdown.parse("""
**Profile likelihood confidence intervals**

| param | true | 95% CI (MAE) | 95% CI (MSE) |
|-------|------|-------------|-------------|
$(join(rows, "\n"))

The MSE CIs are shifted toward the outlier run's inflated K, while the MAE
CIs remain centred on the true parameters. This illustrates why raw replicate
storage matters: you can choose a robust aggregation after the CM runs are done,
without re-running the CM.
""")
end

# ╔═╡ 0000001d-0000-0000-0000-000000000000
md"""
## 8  Sampling SM Predictions
"""

# ╔═╡ 0000001e-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(prob_mae, uq_mae; nSamples = 200, rng = rng_sample)
end

# ╔═╡ 0000001f-0000-0000-0000-000000000000
plot(samples)

# ╔═╡ 00000020-0000-0000-0000-000000000000
md"""
## 9  Summary

To use `ReplicateCMData` with your own CM output:

1. Replace the `runs` array with your replicate trajectories, shaped
   `[n_times, n_variables, n_replicates, n_conditions, n_param_sets]`.
2. Choose or define a loss function:
   - `replicate_mae` — robust to outliers, fits the median run.
   - `replicate_mse` — fits the mean run; equivalent to using `CMData` + `GaussianNLL`
     with `σ = std(runs)` (up to a constant).
   - Any other aggregation: quantile loss, trimmed mean, rank-based, etc.
3. Pass your loss as `SMFitProblem(sm, data, prior; loss = CustomLoss(my_loss))` and then
   call `fitSurrogate(prob, P0)`.

All downstream code — profile likelihood, prediction sampling — works unchanged.
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╠═00000007-0000-0000-0000-000000000000
# ╟─00000008-0000-0000-0000-000000000000
# ╠═00000009-0000-0000-0000-000000000000
# ╠═0000000a-0000-0000-0000-000000000000
# ╠═0000000b-0000-0000-0000-000000000000
# ╟─0000000c-0000-0000-0000-000000000000
# ╠═0000000d-0000-0000-0000-000000000000
# ╟─0000000e-0000-0000-0000-000000000000
# ╠═0000000f-0000-0000-0000-000000000000
# ╟─00000010-0000-0000-0000-000000000000
# ╠═00000011-0000-0000-0000-000000000000
# ╠═00000012-0000-0000-0000-000000000000
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
