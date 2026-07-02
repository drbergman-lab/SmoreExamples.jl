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
#   Pluto.run(notebook                   = "/path/to/single_obs_custom_data.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# Custom CM Data: Single Observation Per Parameter Set

`CMData` stores both a mean array `μ` and a standard deviation array `σ` of
equal shape. This works well when the CM has been run multiple times at each
parameter set — you compute `μ = mean(runs)` and `σ = std(runs)` and pass both
in.

But what if the CM was run only **once** at each parameter set? There are no
replicates, so no empirical `σ` to compute. Noise must instead come from domain
knowledge — for example, a proportional (constant-CV) error model is standard
in systems biology and PK/PD:

$$\sigma(t) = \text{cv} \times |y(t)|$$

where `cv` is a known coefficient of variation (e.g., 0.10 for a 10% assay CV).

Rather than allocating a full `σ` array just to encode this scalar rule, we
define a custom `AbstractCMData` subtype — `SingleObsCMData` — that stores
only the observations and a single `cv` value, and computes `σ` on-the-fly.

This notebook shows the full SmoreBase pipeline (fit → profile likelihood UQ
→ prediction sampling) using this custom type. The logistic growth SM is reused
from `logistic_growth_pipeline.jl` so the two notebooks can be compared
directly.
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

We need three pieces:

1. **`SingleObsCMData`** — the container that holds the raw observations and
   the scalar CV. It subtype of `AbstractCMData`.

2. **`SingleObsCMDataSlice`** — a single-parameter-set view, subtype of
   `AbstractCMDataSlice`. Slices are what the fitting loop actually sees.

3. **Interface methods** — `_sliceCmParamSet` tells SmoreBase how to cut the
   container down to one parameter set; `_mean`, `_sd`, and `_cov` tell the
   default `GaussianNLL` loss where to find each quantity.
"""

# ╔═╡ 00000009-0000-0000-0000-000000000000
struct SingleObsCMData <: AbstractCMData
	obs::Array{Float64,4}                 # [n_times, n_variables, n_conditions, n_cm_param_sets]
	cv::Float64                           # proportional noise level, e.g. 0.10 → 10% CV
	times::Union{Nothing,Vector{Float64}} # time grid (passed straight through to slice)
end

# ╔═╡ 0000000a-0000-0000-0000-000000000000
struct SingleObsCMDataSlice <: AbstractCMDataSlice
	obs::AbstractArray{Float64,3}         # @view into parent [n_times, n_variables, n_conditions]
	cv::Float64
	times::Union{Nothing,Vector{Float64}}
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
begin
	function SmoreBase._sliceCmParamSet(data::SingleObsCMData, pi::Int)
		SingleObsCMDataSlice(@view(data.obs[:, :, :, pi]), data.cv, data.times)
	end

	SmoreBase._times(d::SingleObsCMData)      = d.times
	SmoreBase._times(d::SingleObsCMDataSlice) = d.times

	SmoreBase._mean(d::SingleObsCMDataSlice) = d.obs
	SmoreBase._sd(d::SingleObsCMDataSlice)   = d.cv .* abs.(d.obs)
	SmoreBase._cov(d::SingleObsCMDataSlice)  = nothing

	# Required by fitSurrogate for P0 validation
	SmoreBase.n_cm_param_sets(d::SingleObsCMData) = size(d.obs, 4)
end

# ╔═╡ 0000000c-0000-0000-0000-000000000000
md"""
`_sd` returns `cv × |obs|` computed fresh each call — no σ array is stored.
Because `GaussianNLL` calls `_mean` and `_sd` on the slice, the rest of the
pipeline needs no changes.
"""

# ╔═╡ 0000000d-0000-0000-0000-000000000000
md"""
## 3  CM Data

A single deterministic CM trajectory (one run per parameter set). Here we
generate it synthetically by evaluating the logistic at the true parameters.
"""

# ╔═╡ 0000000e-0000-0000-0000-000000000000
begin
	obs_1d = vec(logistic(t, p_true, nothing))   # [n_times] — one trajectory, no replicates

	data = SingleObsCMData(
		reshape(obs_1d, length(t), 1, 1, 1),     # promote to [n_times, 1, 1, 1]
		0.10,                                    # 10% CV measurement error
		t,
	)
end

# ╔═╡ 0000000f-0000-0000-0000-000000000000
md"Shape: $(size(data.obs, 1)) times × 1 variable × 1 condition × 1 param-set (no σ array stored)"

# ╔═╡ 00000010-0000-0000-0000-000000000000
md"""
## 4  Parameter Prior
"""

# ╔═╡ 00000011-0000-0000-0000-000000000000
prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])

# ╔═╡ 0000001e-0000-0000-0000-000000000000
prob = SMFitProblem(sm, data, prior)

# ╔═╡ 00000012-0000-0000-0000-000000000000
md"""
## 5  Fitting the SM

`SMFitProblem` bundles the SM, data, and prior. `fitSurrogate` then minimises
the Gaussian NLL between the SM prediction and our custom data, calling `_mean`
and `_sd` on each slice — no `loss` keyword needed.
"""

# ╔═╡ 00000013-0000-0000-0000-000000000000
begin
	P0     = [0.5 5.0]
	result = fitSurrogate(prob, P0)
end

# ╔═╡ 00000014-0000-0000-0000-000000000000
md"""
**Fit summary**

| Parameter | Fitted | True | \|error\| |
|-----------|--------|------|-----------|
| r | $(round(result.parameters[1,1]; digits=4)) | $(p_true[1]) | $(round(abs(result.parameters[1,1] - p_true[1]); sigdigits=2)) |
| K | $(round(result.parameters[1,2]; digits=4)) | $(p_true[2]) | $(round(abs(result.parameters[1,2] - p_true[2]); sigdigits=2)) |

Converged: $(result.converged[1]) · NLL at fit: $(round(-result.errors[1]; digits=4))
"""

# ╔═╡ 00000015-0000-0000-0000-000000000000
let
	p_fit   = result.parameters[1, :]
	ŷ       = vec(logistic(t, p_fit, nothing))
	obs_v   = vec(data.obs[:, 1, 1, 1])
	σ_v     = data.cv .* abs.(obs_v)

	plt = plot(; xlabel = "Time", ylabel = "Value",
		title = "SM fit vs single CM observation")
	scatter!(plt, t, obs_v; yerror = σ_v, label = "CM obs ± $(Int(round(data.cv * 100)))% CV")
	plot!(plt, t, ŷ; linewidth = 2, label = "SM fit")
	plt
end

# ╔═╡ 00000016-0000-0000-0000-000000000000
md"""
## 6  Profile Likelihood UQ
"""

# ╔═╡ 00000017-0000-0000-0000-000000000000
uq = quantifyUncertainty(ProfileLikelihood(n_points = 25, confidence_level = 0.95), prob, result, 1)

# ╔═╡ 00000018-0000-0000-0000-000000000000
plot(uq)

# ╔═╡ 00000019-0000-0000-0000-000000000000
let rows = map(uq.profiles) do pc
	fitted = round(result.parameters[1, pc.parameter_index]; digits = 4)
	lo = pc.ci_lower === nothing ? "—" : string(round(pc.ci_lower; digits = 4))
	hi = pc.ci_upper === nothing ? "—" : string(round(pc.ci_upper; digits = 4))
	note = pc.ci_lower === nothing ? " *(profile flat — weakly identified)*" : ""
	"| $(pc.parameter_name) | $fitted | [$lo, $hi] |$note"
end
Markdown.parse("""
**Profile likelihood confidence intervals (proportional error model, cv = $(data.cv))**

| param | MLE | 95% CI |
|-------|-----|--------|
$(join(rows, "\n"))

CIs are wider than they would be with a fixed-σ Gaussian because the noise
level grows with the signal: at early times obs ≈ 0 so the residuals are down-
weighted, while at plateau the noise is largest.
""")
end

# ╔═╡ 0000001a-0000-0000-0000-000000000000
md"""
## 7  Sampling SM Predictions
"""

# ╔═╡ 0000001b-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(prob, uq; nSamples = 200, rng = rng_sample)
end

# ╔═╡ 0000001c-0000-0000-0000-000000000000
plot(samples)

# ╔═╡ 0000001d-0000-0000-0000-000000000000
md"""
## 8  Summary

To plug in your own single-observation CM data:

1. Replace `obs_1d` with your actual CM output array.
2. Set `cv` to the instrument coefficient of variation (or another noise spec).
3. Define `_sd` to return whatever noise model fits your domain.

If a fixed absolute noise floor is more appropriate than a proportional model,
change `_sd` to:

```julia
SmoreBase._sd(d::SingleObsCMDataSlice) = fill(d.cv, size(d.obs))
```

where `cv` now holds an absolute noise level. All downstream code — `SMFitProblem`,
`fitSurrogate`, profile likelihood, prediction sampling — stays unchanged.
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
# ╟─0000000d-0000-0000-0000-000000000000
# ╠═0000000e-0000-0000-0000-000000000000
# ╟─0000000f-0000-0000-0000-000000000000
# ╟─00000010-0000-0000-0000-000000000000
# ╠═00000011-0000-0000-0000-000000000000
# ╠═0000001e-0000-0000-0000-000000000000
# ╟─00000012-0000-0000-0000-000000000000
# ╠═00000013-0000-0000-0000-000000000000
# ╟─00000014-0000-0000-0000-000000000000
# ╠═00000015-0000-0000-0000-000000000000
# ╟─00000016-0000-0000-0000-000000000000
# ╠═00000017-0000-0000-0000-000000000000
# ╠═00000018-0000-0000-0000-000000000000
# ╠═00000019-0000-0000-0000-000000000000
# ╟─0000001a-0000-0000-0000-000000000000
# ╠═0000001b-0000-0000-0000-000000000000
# ╠═0000001c-0000-0000-0000-000000000000
# ╟─0000001d-0000-0000-0000-000000000000
