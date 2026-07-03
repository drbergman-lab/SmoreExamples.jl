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
#   Pluto.run(notebook                   = "/path/to/logistic_growth_pipeline.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# The SmoreVerse Pipeline: Logistic Growth Tutorial

**SmoreVerse** sits between a slow, expensive *complex model* (CM) and the
real world. A fast *surrogate model* (SM) is trained on CM-generated output,
then used as a proxy for fitting to data and for analysing CM behaviour.

This notebook walks through `SmoreBase`'s fit/UQ/sampling pipeline and
`SmoreGSA`'s sensitivity analysis, using **logistic growth** as a toy SM:

| Step | Sub-package | What it does |
|------|-------------|--------------|
| 1–4  | `SmoreBase` | Build CM data, define the SM, fit parameters |
| 5–6  | `SmoreBase` | Quantify SM parameter uncertainty; sample predictions |
| 7    | `SmoreGSA`  | Global sensitivity of SM outputs to CM parameters |

The remaining pipeline step — building a posterior over CM parameters from
real-world data via `SmoreFit` — is in the companion notebook
[`cm_posterior_pipeline.jl`](./cm_posterior_pipeline.jl); see Section 8 below.

The time window runs through the inflection point and into saturation, so both
the growth rate $r$ and the carrying capacity $K$ are identifiable. Another
companion notebook, [`nonidentifiability.jl`](./nonidentifiability.jl), revisits
the model on a shorter, exponential-phase-only window to show what a
*non-identifiable* parameter looks like.

All intermediate results are kept in scope so you can inspect them in the
REPL after running the notebook.
"""

# ╔═╡ 0000000a-0000-0000-0000-000000000000
md"""
## 1  CM Data

In a real SmoreVerse application, CM output would come from an agent-based
model's summary statistics across replicate runs. Here, so results can be
checked against known ground truth, we instead generate synthetic CM data
using logistic growth plus noise:

$$y(t) = \frac{K}{1 + \left(\frac{K}{y_0} - 1\right) e^{-r t}}, \qquad y_0 = 0.01$$

with true parameters $r = 0.6$ and $K = 4.0$. `CMData` holds the mean ($\mu$)
and standard deviation ($\sigma$) of CM output across stochastic replicates,
at each time point; the constructor normalises any input array to the
canonical 4-D shape `[n_times × n_variables × n_conditions × n_cm_param_sets]`.
In a real workflow, $\mu$ and $\sigma$ would come from many CM replicate runs.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
# Implemented as a plain Julia function so it can double as the SM in Section 2 below.
logistic(t, p, _cond) = reshape(
	p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
	:, 1,
)

# ╔═╡ 00000007-0000-0000-0000-000000000000
# Time grid and ground-truth parameters used throughout this notebook.
begin
	t      = collect(0.0:1.0:20.0)  # 21 time points: exponential rise → inflection → saturation
	p_true = [0.6, 4.0]             # true r and K
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
begin
	noise_σ = 0.05
	μ_true  = vec(logistic(t, p_true, nothing))

	data = CMData(
		μ     = μ_true,
		σ     = fill(noise_σ, length(μ_true)),
		times = t,
	)
end

# ╔═╡ 0000000c-0000-0000-0000-000000000000
md"Shape: $(n_times(data)) times × $(n_variables(data)) variables × $(n_conditions(data)) conditions × $(n_cm_param_sets(data)) CM parameter sets"

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 2  The Surrogate Model

Having observed logistic growth in the CM data above, we use a logistic
growth SM — the same function object used above to generate the CM data —
with two free parameters: growth rate $r$ and carrying capacity $K$.

`AnalyticalSurrogateModel` wraps any closed-form function with the signature

```julia
fn(t::Vector, p::Vector, condition::String) -> Matrix{Float64}
```

Rows index time points and columns index output variables. A single-variable
SM must return an `[n_times × 1]` matrix — hence the `reshape` in `logistic`'s
definition above.
"""

# ╔═╡ 00000006-0000-0000-0000-000000000000
sm = AnalyticalSurrogateModel(fn = logistic)

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"Evaluating the SM at the true parameters gives an `[21 × 1]` matrix:"

# ╔═╡ 00000009-0000-0000-0000-000000000000
SmoreBase._evaluate(sm, t, p_true, "default")

# ╔═╡ 0000000d-0000-0000-0000-000000000000
md"""
## 3  Parameter Prior

`ParameterPrior` encodes the feasible region for SM parameters. The default
constructor builds independent `Uniform(lower, upper)` distributions; custom
`Distributions.jl` distributions can also be supplied.

The prior serves two roles:
- **Optimisation bounds** — passed to the bounded LBFGS solver in `fitSurrogate`.
- **Profile sweep range** — defines the grid endpoints for each profile likelihood curve.
"""

# ╔═╡ 0000000e-0000-0000-0000-000000000000
prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])

# ╔═╡ 00000030-0000-0000-0000-000000000000
prob = SMFitProblem(sm, data, prior)

# ╔═╡ 0000000f-0000-0000-0000-000000000000
md"""
## 4  Fitting the SM

`SMFitProblem` bundles the surrogate model, CM data, prior, and (optionally)
a custom loss into a single object. `fitSurrogate` then takes the problem and
an initial guess `P0` — a `[n_cm_param_sets × n_sm_params]` matrix (one row
per CM parameter set), or a single vector broadcast to every CM parameter set
(there is only one here).
"""

# ╔═╡ 00000010-0000-0000-0000-000000000000
begin
	P0     = [0.5 5.0]
	result = fitSurrogate(prob, P0)
end

# ╔═╡ 00000011-0000-0000-0000-000000000000
md"""
**Fit summary**

| Parameter | Fitted | True | \|error\| |
|-----------|--------|------|-----------|
| r | $(round(result.parameters[1,1]; digits=4)) | $(p_true[1]) | $(round(abs(result.parameters[1,1] - p_true[1]); sigdigits=2)) |
| K | $(round(result.parameters[1,2]; digits=4)) | $(p_true[2]) | $(round(abs(result.parameters[1,2] - p_true[2]); sigdigits=2)) |

Converged: $(result.converged[1]) · NLL at fit: $(round(-result.errors[1]; digits=4))
"""

# ╔═╡ 0000002b-0000-0000-0000-000000000000
plot(SMFitPlot(sm, data, result))

# ╔═╡ 00000012-0000-0000-0000-000000000000
md"""
## 5  Profile Likelihood UQ

After fitting we quantify uncertainty using **profile likelihood** (Wilks' theorem).
For each parameter $\theta_i$:

1. Sweep a grid of values across the prior range.
2. At each grid point, **fix** $\theta_i$ and **re-optimise** all remaining parameters.
3. Record the profile log-likelihood $\text{PL}(\theta_i)$.
4. The $95\%$ confidence interval is $\{\theta_i : \text{PL}(\theta_i) \ge L^* - \tfrac{1}{2}\chi^2_{1,0.95}\}$, where $L^*$ is the log-likelihood at the MLE.

This is a **product-measure approximation** — each parameter is profiled
independently, so correlations are not captured.
"""

# ╔═╡ 00000013-0000-0000-0000-000000000000
uq = quantifyUncertainty(ProfileLikelihood(n_points = 25, confidence_level = 0.95), prob, result, 1)

# ╔═╡ 0000002c-0000-0000-0000-000000000000
plot(uq)

# ╔═╡ 00000014-0000-0000-0000-000000000000
let rows = map(uq.profiles) do pc
	fitted = round(result.parameters[1, pc.parameter_index]; digits = 4)
	lo = pc.ci_lower === nothing ? "—" : string(round(pc.ci_lower; digits = 4))
	hi = pc.ci_upper === nothing ? "—" : string(round(pc.ci_upper; digits = 4))
	note = pc.ci_lower === nothing ? " *(profile flat — weakly identified)*" : ""
	"| $(pc.parameter_name) | $fitted | [$lo, $hi] |$note"
end
Markdown.parse("""
**Profile likelihood confidence intervals**

| param | MLE | 95% CI |
|-------|-----|--------|
$(join(rows, "\n"))

Both `r` and `K` are well identified: the time window runs through the
inflection point and into saturation, so the data constrain the carrying
capacity from above. For the contrasting case — data confined to the early
exponential phase, where `K`'s profile stays flat and its CI is unbounded — see
the companion notebook [`nonidentifiability.jl`](./nonidentifiability.jl).
""")
end

# ╔═╡ 00000015-0000-0000-0000-000000000000
md"""
## 6  Sampling SM Predictions

`sampleSMPredictions` draws SM parameter vectors from the uncertainty region
encoded by the `ProfileLikelihoodResult`, then evaluates the SM at each draw,
producing a **prediction envelope**.

Internally it uses a **Sobol low-discrepancy sequence** in
$[0,1]^{n_\text{SM params}}$ with a **Cranley–Patterson random shift** for
reproducibility, then applies the per-parameter profile-LL inverse CDF to map
each unit-cube draw to the natural parameter scale. Where the profile is
narrow (well-identified, as both $r$ and $K$ are on this window), the envelope
is tight; where a profile is flat (weakly identified), the envelope fans out —
see [`nonidentifiability.jl`](./nonidentifiability.jl) for that case.
"""

# ╔═╡ 00000016-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(prob, uq; nSamples = 200, rng = rng_sample)
end

# ╔═╡ 00000017-0000-0000-0000-000000000000
let
	pred_mean = dropdims(mean(samples.predictions; dims = 3); dims = 3)
	pred_lo   = dropdims(minimum(samples.predictions; dims = 3); dims = 3)
	pred_hi   = dropdims(maximum(samples.predictions; dims = 3); dims = 3)

	rows = map(eachindex(t)) do i
		"| $(t[i]) | $(round(μ_true[i]; digits=4)) | $(round(pred_mean[i,1]; digits=4)) | [$(round(pred_lo[i,1]; digits=4)), $(round(pred_hi[i,1]; digits=4))] |"
	end
	Markdown.parse("""
**Prediction envelope  (200 samples)**

| time | true | mean | [min, max] |
|------|------|------|------------|
$(join(rows, "\n"))
""")
end

# ╔═╡ 00000029-0000-0000-0000-000000000000
plot(samples)

# ╔═╡ 00000018-0000-0000-0000-000000000000
md"""
## 7  Global Sensitivity Analysis

`runSensitivity` asks: *how sensitive are the SM's outputs to variation in the
CM parameters?* The CM parameters control the biology (e.g., a cell kill rate
or proliferation rate); as they vary, so do the best-fit SM parameters and
their uncertainty.

Here the two CM parameters map directly onto the logistic SM: `cm_r` maps to the
growth rate $r$, and `cm_K` maps to the carrying capacity $K$. That one-to-one
mapping lets us predict the sensitivities in advance — a handy sanity check on
the GSA machinery:

- the **saturated** output $y(T) \approx K$ should depend almost entirely on `cm_K`;
- an **early**, exponential-phase output should depend almost entirely on `cm_r`.

Section 7a confirms the first (the default output is the last time point) and
Section 7c the second.

The GSA engine (EFAST or Morris) sweeps the CM parameter space by calling a
function $f(u)$, $u \in [0,1]^{n_\text{CM}}$, at many points. That function:

1. Converts $u$ to CM parameters via the `cm_prior` inverse CDF.
2. Interpolates SM parameter CI bounds at that CM point from a pre-built table.
3. Draws SM parameter samples within those bounds (Sobol + Cranley–Patterson).
4. Returns the average `outputFn` value across those samples.

Setting up the GSA therefore requires a **list of `ProfileLikelihoodResult`
objects** — one per CM parameter set — that encode the SM parameter uncertainty at
each CM parameter value where the CM was actually run.
"""

# ╔═╡ 00000019-0000-0000-0000-000000000000
sm_gsa = sm   # the logistic SM from Section 2

# ╔═╡ 0000001a-0000-0000-0000-000000000000
t_gsa = collect(range(0.0, 20.0, 21))   # exponential rise → saturation, as in Section 1

# ╔═╡ 0000001b-0000-0000-0000-000000000000
md"""
### Constructing UQ results for every CM parameter set

GSA needs one `ProfileLikelihoodResult` per CM parameter set — the SM-parameter
uncertainty induced by the CM running at each grid point. `CMData`'s
`cm_param_sets` axis holds all of them at once:

1. Choose a grid of CM parameter values (here: `cm_r ∈ {0.4, 0.6, 0.8}`, `cm_K ∈ {2,…,6}` — 15 CM parameter sets in total).
2. Build one `CMData` whose `cm_param_sets` axis holds all 15 columns. Here
   we generate each column by evaluating the SM at the CM-true parameters; in a
   real workflow these are the CM's own per-parameter-set outputs.
3. `fitSurrogate` and `quantifyUncertainty`, each called once over the whole
   `CMData`, return the `SMFitResult` and `Vector{ProfileLikelihoodResult}` GSA
   needs, row-aligned with the CM parameter sets.

Because the SM is analytic, all 15 fits take only a second or two. When the CM
is genuinely expensive you would precompute and cache this `CMData`, but
the fit + profile calls themselves are unchanged.
"""

# ╔═╡ 0000001d-0000-0000-0000-000000000000
# 3 × 5 = 15 CM param_sets: cm_r ∈ {0.4,0.6,0.8} (sets growth rate r) × cm_K ∈ {2,…,6} (sets carrying capacity K).
begin
	cm_r_vals = [0.4, 0.6, 0.8]
	cm_K_vals = Float64.(2:6)

	cm_sample = GridCMSample(cm_r_vals, cm_K_vals; names = ["cm_r", "cm_K"])
	cm_rs, cm_Ks = cm_sample.params[:, 1], cm_sample.params[:, 2]
	n_ps = length(cm_rs)

	# One CMData holding all 15 CM param_sets (real workflow: 15 CM outputs, not evaluations of the SM).
	μ_cm_param_sets = reduce(hcat, [vec(logistic(t_gsa, [cm_rs[i], cm_Ks[i]], nothing)) for i in 1:n_ps])
	d_cm_param_sets = CMData(μ = μ_cm_param_sets, σ = fill(noise_σ, size(μ_cm_param_sets)), times = t_gsa, cm_param_sets = n_ps)
	prob_cm_param_sets = SMFitProblem(sm, d_cm_param_sets, prior)

	fit_cm_param_sets = fitSurrogate(prob_cm_param_sets, [cm_rs cm_Ks])   # one row per CM param_set
	uq_list = quantifyUncertainty(ProfileLikelihood(n_points = 15, confidence_level = 0.95), prob_cm_param_sets, fit_cm_param_sets)

	cm_prior = ParameterPrior([0.4, 2.0], [0.8, 6.0]; names = ["cm_r", "cm_K"])
end

# ╔═╡ 0000001e-0000-0000-0000-000000000000
let fmt(x) = x === nothing ? "—" : string(round(x; digits = 3))
	ci(pc) = "[$(fmt(pc.ci_lower)), $(fmt(pc.ci_upper))]"
	rows = map(eachindex(cm_rs)) do i
		uq_i = uq_list[i]
		# fit_result is the shared SMFitResult across all CM param_sets; row i is this one's fit.
		r = round(uq_i.fit_result.parameters[i, 1]; digits = 3)
		K = round(uq_i.fit_result.parameters[i, 2]; digits = 3)
		"| $(cm_rs[i]) | $(cm_Ks[i]) | $r | $(ci(uq_i.profiles[1])) | $K | $(ci(uq_i.profiles[2])) |"
	end
Markdown.parse("""
**CM parameter set summary** ($(length(cm_rs)) CM parameter sets)

| cm\\_r | cm\\_K | r (fit) | r CI | K (fit) | K CI |
|--------|--------|---------|------|---------|------|
$(join(rows, "\n"))
""")
end

# ╔═╡ 0000001f-0000-0000-0000-000000000000
md"""
### 7a  EFAST

EFAST decomposes output variance into contributions from each CM parameter:

- **S1** — first-order index: fraction of variance explained by `cm_r` alone.
- **ST** — total-order index: fraction including all interactions involving `cm_r`.

The default `outputFn` summarises each prediction by its **last** time point.
Because our window runs out to saturation, that value is $y(T) \approx K$, so we
expect `cm_K` to dominate the variance and `cm_r` to contribute very little. The
complementary picture — `cm_r` dominating an *early*-time output — appears in
Section 7c.
"""

# ╔═╡ 00000020-0000-0000-0000-000000000000
begin
	rng_efast    = Random.MersenneTwister(42)
	result_efast = runSensitivity(
		EFAST(n_samples = 100), sm_gsa, uq_list, cm_sample, cm_prior;
		times = t_gsa,
		rng   = rng_efast,
	)
end

# ╔═╡ 00000021-0000-0000-0000-000000000000
let S1 = sensitivity_S1(result_efast), ST = sensitivity_ST(result_efast)
	rows = map(eachindex(result_efast.cm_parameter_names)) do j
		"| $(result_efast.cm_parameter_names[j]) | $(round(S1[1,j]; digits=4)) | $(round(ST[1,j]; digits=4)) |"
	end
	Markdown.parse("""
**EFAST results**  (output: `$(result_efast.output_labels[1])`)

| CM param | S1 | ST |
|----------|----|----|
$(join(rows, "\n"))
""")
end

# ╔═╡ 0000002d-0000-0000-0000-000000000000
plot(result_efast)

# ╔═╡ 00000022-0000-0000-0000-000000000000
md"""
### 7b  Morris Screening

Morris' elementary-effects method is a cheaper screening alternative. It
returns $\mu^*$ (mean absolute elementary effect) — a measure of importance —
but not total-order indices. Use it to rank parameters before committing to a
full EFAST run.
"""

# ╔═╡ 00000023-0000-0000-0000-000000000000
begin
	rng_morris    = Random.MersenneTwister(7)
	result_morris = runSensitivity(
		Morris(num_trajectory = 10), sm_gsa, uq_list, cm_sample, cm_prior;
		times = t_gsa,
		rng   = rng_morris,
	)
end

# ╔═╡ 00000024-0000-0000-0000-000000000000
sensitivity_S1(result_morris)   # μ* — [n_outputs × n_cm_params]

# ╔═╡ 0000002e-0000-0000-0000-000000000000
plot(result_morris)

# ╔═╡ 00000025-0000-0000-0000-000000000000
md"""
### 7c  Custom `outputFn`

By default `outputFn` extracts the last time-point value. Supplying a custom
function lets you compute any scalar summary — here we extract an **early** and
the **final** (saturated) time point separately. We define the early point as
the time at which the trajectory reaches **10% of carrying capacity** (found by
solving $y(t) = 0.1\,K$), which keeps it deep in the exponential phase where
growth is governed by the rate — so the early output is sensitive almost
entirely to `cm_r`. At saturation $y(T) \approx K$, so the final output is
sensitive almost entirely to `cm_K`. This cleanly separates the two parameters'
regimes.
"""

# ╔═╡ 00000026-0000-0000-0000-000000000000
# "Early" output: the time at which the nominal trajectory reaches 10% of K.
# Solving y(t) = 0.1·K for t keeps the point in the exponential phase and is
# robust to changes in the time grid.
begin
	early_frac        = 0.1
	t_early           = log(early_frac * (p_true[2] / 0.01 - 1) / (1 - early_frac)) / p_true[1]
	i_early           = argmin(abs.(t_gsa .- t_early))
	two_outputs(pred) = [pred[i_early, 1], pred[end, 1]]
end

# ╔═╡ 00000027-0000-0000-0000-000000000000
begin
	rng_custom    = Random.MersenneTwister(99)
	result_custom = runSensitivity(
		EFAST(n_samples = 100), sm_gsa, uq_list, cm_sample, cm_prior;
		times    = t_gsa,
		outputFn = two_outputs,
		rng      = rng_custom,
	)
end

# ╔═╡ 00000028-0000-0000-0000-000000000000
let S1 = sensitivity_S1(result_custom), ST = sensitivity_ST(result_custom)
	output_labels = ["t = $(t_gsa[i_early])  (≈10% of K, exponential phase)", "t = $(t_gsa[end])  (saturated)"]
	pnames = result_custom.cm_parameter_names
	header = "| output |" * join([" S1($(p)) | ST($(p)) |" for p in pnames], "")
	sep    = "|--------|" * repeat("----------|----------|", length(pnames))
	rows = map(1:2) do i
		cells = join(["$(round(S1[i,j]; digits=4)) | $(round(ST[i,j]; digits=4))" for j in eachindex(pnames)], " | ")
		"| $(output_labels[i]) | $cells |"
	end
	Markdown.parse("""
**EFAST — two outputs, two CM parameters**

$header
$sep
$(join(rows, "\n"))
""")
end

# ╔═╡ 0000002f-0000-0000-0000-000000000000
plot(result_custom)

# ╔═╡ 00000031-0000-0000-0000-000000000000
md"""
## 8  Next: CM Posterior from Real Data

GSA tells you which CM parameters *matter*. The complementary question —
given **real-world observations**, which CM parameter values are *consistent*
with that data? — is answered by `SmoreFit.buildPosterior`, which works from the
same per-CM-parameter-set `ProfileLikelihoodResult` list that GSA consumes. See
[`cm_posterior_pipeline.jl`](./cm_posterior_pipeline.jl) for the full Step 8
walk-through (bridge methods, accept/graded posteriors, interior queries).
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─0000000a-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000007-0000-0000-0000-000000000000
# ╠═0000000b-0000-0000-0000-000000000000
# ╟─0000000c-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╟─00000008-0000-0000-0000-000000000000
# ╠═00000009-0000-0000-0000-000000000000
# ╟─0000000d-0000-0000-0000-000000000000
# ╠═0000000e-0000-0000-0000-000000000000
# ╠═00000030-0000-0000-0000-000000000000
# ╟─0000000f-0000-0000-0000-000000000000
# ╠═00000010-0000-0000-0000-000000000000
# ╟─00000011-0000-0000-0000-000000000000
# ╠═0000002b-0000-0000-0000-000000000000
# ╟─00000012-0000-0000-0000-000000000000
# ╠═00000013-0000-0000-0000-000000000000
# ╠═0000002c-0000-0000-0000-000000000000
# ╠═00000014-0000-0000-0000-000000000000
# ╟─00000015-0000-0000-0000-000000000000
# ╠═00000016-0000-0000-0000-000000000000
# ╠═00000017-0000-0000-0000-000000000000
# ╠═00000029-0000-0000-0000-000000000000
# ╟─00000018-0000-0000-0000-000000000000
# ╠═00000019-0000-0000-0000-000000000000
# ╠═0000001a-0000-0000-0000-000000000000
# ╟─0000001b-0000-0000-0000-000000000000
# ╠═0000001d-0000-0000-0000-000000000000
# ╠═0000001e-0000-0000-0000-000000000000
# ╟─0000001f-0000-0000-0000-000000000000
# ╠═00000020-0000-0000-0000-000000000000
# ╠═00000021-0000-0000-0000-000000000000
# ╠═0000002d-0000-0000-0000-000000000000
# ╟─00000022-0000-0000-0000-000000000000
# ╠═00000023-0000-0000-0000-000000000000
# ╠═00000024-0000-0000-0000-000000000000
# ╠═0000002e-0000-0000-0000-000000000000
# ╟─00000025-0000-0000-0000-000000000000
# ╠═00000026-0000-0000-0000-000000000000
# ╠═00000027-0000-0000-0000-000000000000
# ╠═00000028-0000-0000-0000-000000000000
# ╠═0000002f-0000-0000-0000-000000000000
# ╟─00000031-0000-0000-0000-000000000000
