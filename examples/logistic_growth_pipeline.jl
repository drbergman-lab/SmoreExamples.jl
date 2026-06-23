### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000000
# Launch via SmoreExamples.run_example(), or manually:
#
#   using Pluto
#   Pluto.run(notebook                   = "/path/to/logistic_growth_pipeline.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000002-0000-0000-0000-000000000000
begin
	using Smore
	using CairoMakie
	using Distributions
	using Random
	using Statistics
end

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# The SmoreVerse Pipeline: Logistic Growth Tutorial

**SmoreVerse** sits between a slow, expensive *complex model* (CM) and the
real world. A fast *surrogate model* (SM) is trained on CM-generated output,
then used as a proxy for fitting to data and for analysing CM behaviour.

This notebook walks through the full pipeline using **logistic growth** as a
toy SM, covering three sub-packages in sequence:

| Step | Sub-package | What it does |
|------|-------------|--------------|
| 1–4  | `SmoreBase` | Define the SM, build data, fit parameters |
| 5–6  | `SmoreBase` | Quantify SM parameter uncertainty; sample predictions |
| 7    | `SmoreGSA`  | Global sensitivity of SM outputs to CM parameters |

The time window runs through the inflection point and into saturation, so both
the growth rate $r$ and the carrying capacity $K$ are identifiable. A companion
notebook, [`nonidentifiability.jl`](./nonidentifiability.jl), revisits the model
on a shorter, exponential-phase-only window to show what a *non-identifiable*
parameter looks like.

All intermediate results are kept in scope so you can inspect them in the
REPL after running the notebook.
"""

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 1  The Surrogate Model

The SM is logistic growth:

$$y(t) = \frac{K}{1 + \left(\frac{K}{y_0} - 1\right) e^{-r t}}, \qquad y_0 = 0.01$$

with two free parameters: growth rate $r$ and carrying capacity $K$.

In a real SmoreVerse application the SM would be fit to summary statistics
produced by an agent-based model or ODE system (the CM). Here we treat the
logistic equation itself as the SM so results can be checked against known
ground truth.

`AnalyticalSurrogateModel` wraps any closed-form function with the signature

```julia
fn(t::Vector, p::Vector, condition::String) -> Matrix{Float64}
```

Rows index time points and columns index output variables. A single-variable
SM must return an `[n_times × 1]` matrix — hence the `reshape`.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
logistic(t, p, _cond) = reshape(
	p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
	:, 1,
)

# ╔═╡ 00000006-0000-0000-0000-000000000000
sm = AnalyticalSurrogateModel(fn = logistic)

# ╔═╡ 00000007-0000-0000-0000-000000000000
# Time grid and ground-truth parameters used throughout this notebook.
begin
	t      = collect(0.0:1.0:20.0)  # 21 time points: exponential rise → inflection → saturation
	p_true = [0.6, 4.0]             # true r and K
end

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"Evaluating the SM at the true parameters gives an `[21 × 1]` matrix:"

# ╔═╡ 00000009-0000-0000-0000-000000000000
SmoreBase._evaluate(sm, t, p_true, "default")

# ╔═╡ 0000000a-0000-0000-0000-000000000000
md"""
## 2  CM Data

`CMData` holds the mean ($\mu$) and standard deviation ($\sigma$) of CM
output across stochastic replicates, at each time point. The constructor
normalises any input array to the canonical 4-D shape

$$[\text{n\_times} \times \text{n\_variables} \times \text{n\_conditions} \times \text{n\_param\_sets}]$$

Here we generate synthetic CM output by evaluating the logistic at the true
parameters. In a real workflow $\mu$ and $\sigma$ would come from many CM
replicate runs.
"""

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
md"Shape: $(n_times(data)) times × $(n_variables(data)) variables × $(n_conditions(data)) conditions × $(n_param_sets(data)) param\_sets"

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
an initial-guess matrix `P0` `[n_param_sets × n_sm_params]`.
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
uq = quantifyUncertainty(prob, result, ProfileLikelihood(n_points = 25, confidence_level = 0.95))

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
objects** — one per CM cohort — that encode the SM parameter uncertainty at
each CM parameter value where the CM was actually run.
"""

# ╔═╡ 00000019-0000-0000-0000-000000000000
sm_gsa = sm   # the logistic SM from Section 1

# ╔═╡ 0000001a-0000-0000-0000-000000000000
t_gsa = collect(range(0.0, 20.0, 21))   # exponential rise → saturation, as in Section 1

# ╔═╡ 0000001b-0000-0000-0000-000000000000
md"""
### Constructing cohort UQ results

GSA needs one `ProfileLikelihoodResult` per CM cohort — the SM-parameter
uncertainty induced by the CM running at each grid point. We build them with the
very same fit + profile-likelihood workflow from Sections 4–5, now run once per
cohort:

1. Choose a grid of CM parameter values (here: `cm_r ∈ {0.4, 0.6, 0.8}`, `cm_K ∈ {2,…,6}` — 15 cohorts in total).
2. At each grid point, form the cohort's `CMData`. Here we generate it by
   evaluating the SM at the CM-true parameters; in a real workflow this is the
   CM's own output.
3. `fitSurrogate` → `SMFitResult`, then `quantifyUncertainty` → `ProfileLikelihoodResult`.

Because the SM is analytic, all 15 fits take only a second or two. When the CM
is genuinely expensive you would precompute and cache these per-cohort results,
but the GSA call itself is unchanged.
"""

# ╔═╡ 0000001c-0000-0000-0000-000000000000
# Real per-cohort UQ: the Section 4–5 workflow (fit + profile likelihood),
# run on the CMData a single cohort would produce.
function cohort_uq(cm_r, cm_K)
	μ = vec(logistic(t_gsa, [cm_r, cm_K], nothing))
	d = CMData(μ = μ, σ = fill(noise_σ, length(μ)), times = t_gsa)
	p = SMFitProblem(sm, d, prior)
	res = fitSurrogate(p, [cm_r cm_K])
	return quantifyUncertainty(p, res, ProfileLikelihood(n_points = 15, confidence_level = 0.95))
end

# ╔═╡ 0000001d-0000-0000-0000-000000000000
# 3 × 5 = 15 cohorts: cm_r ∈ {0.4,0.6,0.8} (sets growth rate r) × cm_K ∈ {2,…,6} (sets carrying capacity K).
begin
	cm_r_vals = [0.4, 0.6, 0.8]
	cm_K_vals = Float64.(2:6)

	cm_rs        = repeat(cm_r_vals; inner = length(cm_K_vals))
	cm_Ks        = repeat(cm_K_vals; outer = length(cm_r_vals))

	uq_list   = [cohort_uq(r, K) for (r, K) in zip(cm_rs, cm_Ks)]
	cm_sample = GridCMSample([cm_rs cm_Ks])
	cm_prior  = ParameterPrior([0.4, 2.0], [0.8, 6.0]; names = ["cm_r", "cm_K"])
end

# ╔═╡ 0000001e-0000-0000-0000-000000000000
let fmt(x) = x === nothing ? "—" : string(round(x; digits = 3))
	ci(pc) = "[$(fmt(pc.ci_lower)), $(fmt(pc.ci_upper))]"
	rows = map(eachindex(cm_rs)) do i
		uq_i = uq_list[i]
		r = round(uq_i.fit_result.parameters[1, 1]; digits = 3)
		K = round(uq_i.fit_result.parameters[1, 2]; digits = 3)
		"| $(cm_rs[i]) | $(cm_Ks[i]) | $r | $(ci(uq_i.profiles[1])) | $K | $(ci(uq_i.profiles[2])) |"
	end
Markdown.parse("""
**Cohort summary** ($(length(cm_rs)) cohorts)

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
		sm_gsa, uq_list, cm_sample, cm_prior, EFAST(n_samples = 100);
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
		sm_gsa, uq_list, cm_sample, cm_prior, Morris(num_trajectory = 10);
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
		sm_gsa, uq_list, cm_sample, cm_prior, EFAST(n_samples = 100);
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
same kind of per-cohort `ProfileLikelihoodResult` that GSA consumes. See
[`cm_posterior_pipeline.jl`](./cm_posterior_pipeline.jl) for the full Step 8
walk-through (bridge methods, accept/graded posteriors, interior queries).
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
# ╟─0000000a-0000-0000-0000-000000000000
# ╠═0000000b-0000-0000-0000-000000000000
# ╟─0000000c-0000-0000-0000-000000000000
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
# ╠═0000001c-0000-0000-0000-000000000000
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
