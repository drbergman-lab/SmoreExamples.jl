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

This notebook shows how to write your own `AbstractCMData` subtype for feeding SmoreBase's
fitting and UQ pipeline data that doesn't fit the built-in `CMData` container.

The motivating scenario: the CM was run only **once** at each parameter set, producing a single
trajectory with no replicates — so there's no empirical `σ` to compute the way
`CMData` expects. Noise instead has to come from domain knowledge, e.g. a proportional
(constant-CV) error model standard in systems biology and PK/PD:

$$\sigma(t) = \text{cv} \times |y(t)|$$

where `cv` is a known coefficient of variation (e.g., 0.10 for a 10% assay CV).
`SingleObsCMData`, defined below, stores only the raw observations and a single `cv` value,
computing `σ` on-the-fly whenever SmoreBase's fitting/UQ code asks for it — the same pattern
applies to any custom noise model.

The logistic growth SM (reused from `logistic_growth_pipeline.jl`, and standing in here for the
CM as well — it generates the single synthetic trajectory below) drives the rest of the
pipeline — fit → profile likelihood UQ → prediction sampling — unchanged.
"""

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"""
## 1  The Custom Data Type

We need three pieces:

1. **`SingleObsCMData`** — the container that holds the raw observations and
   the scalar CV. It is a subtype of `AbstractCMData`.

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
## 2  CM Data

A single deterministic CM trajectory (one run per parameter set). Here we
generate it synthetically by evaluating the logistic growth function — introduced next as the
SM — at the true parameters.
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

To write your own custom CM data type:

1. Define a container (subtype `AbstractCMData`) that stores your CM data.
2. Define a matching slice type (subtype `AbstractCMDataSlice`) for one parameter set — the view
   the fitting loop actually sees.
3. Implement the interface for a loss function:
    a. If using `GaussianNLL`: `_sliceCmParamSet` (container → slice), `_mean` / `_sd` / `_cov`
           (slice → the quantities `GaussianNLL` needs), and `n_cm_param_sets`.
    b. If using a custom loss: implement whatever methods your loss needs to extract the data.

Once these methods are defined, everything downstream — `SMFitProblem`, `fitSurrogate`, profile
likelihood, prediction sampling — works unchanged; SmoreBase never touches your struct's
internals directly.

```julia
SmoreBase._sd(d::SingleObsCMDataSlice) = fill(d.cv, size(d.obs))
```
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
# ╟─0000000d-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000007-0000-0000-0000-000000000000
# ╠═0000000e-0000-0000-0000-000000000000
# ╟─0000000f-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
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
