### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000000
# Launch Pluto pointing at the SmoreExamples project so that the local
# SmoreBase and SmoreGloS packages resolve correctly:
#
#   using Pluto
#   Pluto.run(project = "/path/to/SmoreExamples")
#
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	Pkg.instantiate()
end

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
| 7    | `SmoreGloS` | Global sensitivity of SM outputs to CM parameters |

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
	t      = collect(0.0:0.5:5.0)   # 11 time points
	p_true = [0.6, 4.0]             # true r and K
end

# ╔═╡ 00000008-0000-0000-0000-000000000000
md"Evaluating the SM at the true parameters gives an `[11 × 1]` matrix:"

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

# ╔═╡ 0000000f-0000-0000-0000-000000000000
md"""
## 4  Fitting the SM

`fitSurrogate` minimises the Gaussian negative log-likelihood between the SM
prediction and the CM summary statistics, using a bounded L-BFGS optimiser.
`P0` is the initial guess matrix `[n_param_sets × n_sm_params]`.
"""

# ╔═╡ 00000010-0000-0000-0000-000000000000
begin
	P0     = [0.5 5.0]
	result = fitSurrogate(sm, data, P0, prior)
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
uq = SmoreBase._uq(sm, data, result, ProfileLikelihood(n_points = 25, confidence_level = 0.95))

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

`K` has no upper CI bound because the data span only the exponential phase of
logistic growth — the carrying capacity is not yet "felt" and the profile
remains flat above the fitted value.
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
narrow (well-identified), the envelope is tight; where it is flat (weakly
identified, like $K$ here), the envelope is wide.
"""

# ╔═╡ 00000016-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(sm, uq; nSamples = 200, rng = rng_sample)
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

# ╔═╡ 0000002a-0000-0000-0000-000000000000
md"""
### GSA surrogate model

The GSA example uses a simpler **exponential decay** SM (distinct from the logistic SM above):

$$y(t) = a \, e^{-b\,t}$$

with amplitude $a$ and decay rate $b$. Two **CM parameters** drive this SM:
`cm_a` and `cm_b`, which will be linked to $a$ and $b$, respectively. Because
of this mapping from CM → SM parameters, we can predict sensitivity results
in advance — a useful sanity check on the GSA machinery.
"""

# ╔═╡ 00000019-0000-0000-0000-000000000000
sm_gsa = AnalyticalSurrogateModel(
	fn = (t, p, _c) -> begin
		a, b = p
		reshape(a .* exp.(-b .* t), :, 1)
	end,
)

# ╔═╡ 0000001a-0000-0000-0000-000000000000
t_gsa = collect(range(0.0, 5.0, 10))

# ╔═╡ 0000001b-0000-0000-0000-000000000000
md"""
### Constructing cohort UQ results

In a complete SmoreVerse workflow you would:

1. Choose a grid of CM parameter values (here: `cm_a ∈ {1,…,5}`, `cm_b ∈ {0.3, 0.5, 0.7}` — 15 cohorts in total).
2. Run the CM at each grid point to obtain summary statistics $(\mu, \sigma)$.
3. Call `fitSurrogate` on each cohort's `CMData` to get an `SMFitResult`.
4. Call `_uq` on each fit to get a `ProfileLikelihoodResult` with a real
   profile LL curve.

**Here we skip steps 1–4** and construct the `ProfileLikelihoodResult` objects
synthetically. This isolates the GSA machinery from the CM execution cost, but
it is important to understand what the synthetic construction approximates and
where it diverges from a real result.

#### What `make_uq` constructs

`make_uq(a_true, b_true)` builds the data structures that would normally emerge
from `fitSurrogate` + `_uq`:

**`SMFitResult`** records the output of `fitSurrogate`. In a real result its
fields carry the best-fit parameters, the NLL at the optimum, convergence
flags, and the raw solver objects. Here we hard-code `parameters = [a_true b_true]`
(the "truth" we would have recovered), `errors = [0.0]` (perfect fit), and
`optim_results = [nothing]`.

**`ProfileCurve`** records the profile likelihood sweep for one SM parameter.
Its key fields are:
- `profile_values` — the grid of swept values (here 20 points over `[0, 5]`).
- `log_likelihoods` — the profile LL at each grid point. **We set these to
  `zeros(20)`, a completely flat profile.** In a real result the curve peaks
  at the MLE and drops off on both sides; `_sampleSMParams` uses `exp(ll)`
  as sampling weights, so a peaked profile concentrates draws near the MLE.
  A flat profile gives `exp(0) = 1` everywhere — uniform weight — so sampling
  falls back to approximately uniform within the CI box. This is a conservative
  approximation: samples are spread more broadly than a peaked profile would allow.
- `optimal_parameters` — the re-optimised full parameter vector at each grid
  point (shape `[n_points × n_sm_params]`). We set this to `zeros(20, 2)` as
  a structural placeholder; the GSA path does not use it.
- `ci_lower`, `ci_upper` — the confidence interval bounds. We set these to
  `±20%` of the true parameter value. In a real result they are determined by
  where the profile LL crosses the Wilks threshold. These bounds are what
  `_buildCMCallable` interpolates across the CM grid to sample SM parameters
  at each GSA evaluation point.

#### Impact on GSA results

The GSA will still correctly *rank* CM parameter importance. The sensitivity
indices may differ quantitatively from what real UQ results would give because:
- Flat profiles → wider, more uniform sampling → different variance decomposition.
- Real profiles concentrate mass near the MLE → tighter sampling → potentially
  higher S1 and lower sampling noise.

In practice, constructing synthetic UQ objects is a useful way to prototype the
GSA pipeline before committing to the computational cost of running the CM.
"""

# ╔═╡ 0000001c-0000-0000-0000-000000000000
function make_uq(a_true, b_true; ci_frac = 0.2)
	lb  = [0.0, 0.0]
	ub  = [5.0, 5.0]
	pr  = ParameterPrior(lb, ub; names = ["a", "b"])
	par = [a_true b_true]
	fit = SMFitResult(par, [0.0], par, pr, BitVector([true]), Any[nothing])

	# ProfileCurve(index, name, profile_values, log_likelihoods, optimal_parameters,
	#              ci_lower, ci_upper, threshold, reference_ll)
	# log_likelihoods = zeros(20): flat profile → uniform sampling within CI box.
	# optimal_parameters = zeros(20,2): placeholder, not used in GSA path.
	pc_a = ProfileCurve(1, "a",
		collect(range(lb[1], ub[1], 20)), zeros(20), zeros(20, 2),
		a_true * (1 - ci_frac), a_true * (1 + ci_frac), -1.92, 0.0)
	pc_b = ProfileCurve(2, "b",
		collect(range(lb[2], ub[2], 20)), zeros(20), zeros(20, 2),
		b_true * (1 - ci_frac), b_true * (1 + ci_frac), -1.92, 0.0)

	return ProfileLikelihoodResult([pc_a, pc_b], fit, t_gsa)
end

# ╔═╡ 0000001d-0000-0000-0000-000000000000
# 5 × 3 = 15 cohorts: cm_a ∈ {1,…,5} (scales amplitude a) × cm_b ∈ {0.3,0.5,0.7} (sets decay rate b).
begin
	cm_a_vals = Float64.(1:5)
	cm_b_vals = [0.3, 0.5, 0.7]

	cm_as        = repeat(cm_a_vals; inner = length(cm_b_vals))
	cm_bs        = repeat(cm_b_vals; outer = length(cm_a_vals))

	uq_list   = [make_uq(a, b) for (a, b) in zip(cm_as, cm_bs)]
	cm_sample = GridCMSample([cm_as cm_bs])
	cm_prior  = ParameterPrior([1.0, 0.3], [5.0, 0.7]; names = ["cm_a", "cm_b"])
end

# ╔═╡ 0000001e-0000-0000-0000-000000000000
let rows = map(eachindex(cm_as)) do i
	uq_i = uq_list[i]
	a = uq_i.fit_result.parameters[1, 1]
	b = uq_i.fit_result.parameters[1, 2]
	"| $(cm_as[i]) | $(cm_bs[i]) | $a | [$(uq_i.profiles[1].ci_lower), $(uq_i.profiles[1].ci_upper)] | $b | [$(uq_i.profiles[2].ci_lower), $(uq_i.profiles[2].ci_upper)] |"
end
Markdown.parse("""
**Cohort summary** ($(length(cm_as)) cohorts)

| cm\\_a | cm\\_b | a (fit) | a CI | b (fit) | b CI |
|--------|--------|---------|------|---------|------|
$(join(rows, "\n"))
""")
end

# ╔═╡ 0000001f-0000-0000-0000-000000000000
md"""
### 7a  EFAST

EFAST decomposes output variance into contributions from each CM parameter:

- **S1** — first-order index: fraction of variance explained by `cm_a` alone.
- **ST** — total-order index: fraction including all interactions involving `cm_a`.

Because `cm_a` directly scales the SM amplitude $a$ and `cm_b` directly sets the decay rate $b$, we expect both to show non-zero sensitivity — with relative magnitudes depending on the time window and the spread in each parameter's prior.
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
function lets you compute any scalar summary — here we extract the **first**
and **last** time points separately. Because $y(0) = a$ is independent of $b$,
the initial output is sensitive only to `cm_a`; the final output
$y(T) = a\,e^{-bT}$ is sensitive to both, with `cm_b` contributing more than
at $t = 0$.
"""

# ╔═╡ 00000026-0000-0000-0000-000000000000
two_outputs(pred) = [pred[1, 1], pred[end, 1]]

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
	output_labels = ["t = $(t_gsa[1])  (initial)", "t = $(t_gsa[end])  (final)"]
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
# ╟─0000002a-0000-0000-0000-000000000000
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
