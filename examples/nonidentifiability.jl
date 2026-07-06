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
# Launch via SmoreExamples.run_example("nonidentifiability.jl"), or manually:
#
#   using Pluto
#   Pluto.run(notebook                   = "/path/to/nonidentifiability.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# Non-Identifiability: Logistic Growth in the Exponential Phase

A parameter is **identifiable** when the data constrain it. When they do not,
its profile likelihood stays **flat**, the confidence interval is unbounded, and
predictions fan out in the unconstrained direction. Recognising this pattern is
essential — a tight point estimate with an unbounded CI is a warning, not a
result.

This notebook reuses the logistic SM from the main pipeline,

$$y(t) = \frac{K}{1 + \left(\frac{K}{y_0} - 1\right) e^{-r t}}, \qquad y_0 = 0.01,$$

but observes it on a **short time window confined to the early, exponential
phase**. There the carrying capacity $K$ is never "felt", so the growth rate $r$
is well identified while $K$ is not.

The full-window companion, `logistic_growth_pipeline.jl` (open it with
`SmoreExamples.run_example("logistic_growth_pipeline.jl")`), runs the *same*
model out to saturation, where both parameters are identifiable — compare the
profile-likelihood and prediction-envelope sections side by side.
"""

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 1  CM Data

We build synthetic CM data by evaluating logistic growth — introduced next as the SM — at known
true parameters, then sampling it only on `t ∈ [0, 5]` — the curve reaches just $y \approx 0.19$
there, far below the true carrying capacity $K = 4$. Twenty-one time points are used, matching
the count in the full-window notebook so the *only* difference is the time horizon.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
# Implemented as a plain Julia function so it can double as the SM in Section 2 below.
logistic(t, p, _cond) = reshape(
	p[2] ./ (1.0 .+ (p[2] / 0.01 - 1.0) .* exp.(-p[1] .* t)),
	:, 1,
)

# ╔═╡ 00000007-0000-0000-0000-000000000000
# Short window: exponential phase only — the carrying capacity is never approached.
begin
	t      = collect(0.0:0.25:5.0)   # 21 time points, all below the inflection
	p_true = [0.6, 4.0]              # true r and K
end

# ╔═╡ 00000008-0000-0000-0000-000000000000
begin
	noise_σ = 0.05
	μ_true  = vec(logistic(t, p_true, nothing))

	data = CMData(
		μ     = μ_true,
		σ     = fill(noise_σ, length(μ_true)),
		times = t,
	)
end

# ╔═╡ 00000017-0000-0000-0000-000000000000
md"""
## 2  The Surrogate Model

Having observed logistic growth in the CM data above, we use a logistic growth SM — the same
function object used above to generate the CM data — same as in `logistic_growth_pipeline.jl`.
"""

# ╔═╡ 00000006-0000-0000-0000-000000000000
sm = CustomSurrogateModel(fn = logistic)

# ╔═╡ 00000009-0000-0000-0000-000000000000
prior = ParameterPrior([0.01, 0.5], [2.0, 10.0]; names = ["r", "K"])

# ╔═╡ 0000000a-0000-0000-0000-000000000000
prob = SMFitProblem(sm, data, prior)

# ╔═╡ 0000000b-0000-0000-0000-000000000000
md"""
## 3  Fit

`fitSurrogate` still returns a point estimate for both parameters. The growth
rate is recovered accurately; the carrying-capacity estimate is whatever value
the optimiser happens to settle on, because the likelihood barely changes as $K$
varies. **A point estimate alone cannot reveal this — we need the UQ step below.**
"""

# ╔═╡ 0000000c-0000-0000-0000-000000000000
begin
	P0     = [0.5 5.0]
	result = fitSurrogate(prob, P0)
end

# ╔═╡ 0000000d-0000-0000-0000-000000000000
md"""
**Fit summary**

| Parameter | Fitted | True | \|error\| |
|-----------|--------|------|-----------|
| r | $(round(result.parameters[1,1]; digits=4)) | $(p_true[1]) | $(round(abs(result.parameters[1,1] - p_true[1]); sigdigits=2)) |
| K | $(round(result.parameters[1,2]; digits=4)) | $(p_true[2]) | $(round(abs(result.parameters[1,2] - p_true[2]); sigdigits=2)) |

Converged: $(result.converged[1]) · NLL at fit: $(round(-result.errors[1]; digits=4))
"""

# ╔═╡ 0000000e-0000-0000-0000-000000000000
plot(SMFitPlot(sm, data, result))

# ╔═╡ 0000000f-0000-0000-0000-000000000000
md"""
## 4  Profile likelihood reveals the non-identifiability

We profile each parameter (fix it on a grid, re-optimise the rest, record the
profile log-likelihood). For an identifiable parameter the profile peaks at the
MLE and crosses the Wilks threshold on both sides, giving a finite CI. For a
non-identifiable one it stays flat, and no threshold crossing exists — the
returned CI bound is `nothing`.
"""

# ╔═╡ 00000010-0000-0000-0000-000000000000
uq = quantifyUncertainty(ProfileLikelihood(n_points = 25, confidence_level = 0.95), prob, result, 1)

# ╔═╡ 00000011-0000-0000-0000-000000000000
plot(uq)

# ╔═╡ 00000012-0000-0000-0000-000000000000
let rows = map(uq.profiles) do pc
	fitted = round(result.parameters[1, pc.parameter_index]; digits = 4)
	lo = pc.ci_lower === nothing ? "—" : string(round(pc.ci_lower; digits = 4))
	hi = pc.ci_upper === nothing ? "—" : string(round(pc.ci_upper; digits = 4))
	note = pc.ci_lower === nothing || pc.ci_upper === nothing ? " *(profile flat — not identified)*" : ""
	"| $(pc.parameter_name) | $fitted | [$lo, $hi] |$note"
end
Markdown.parse("""
**Profile likelihood confidence intervals**

| param | MLE | 95% CI |
|-------|-----|--------|
$(join(rows, "\n"))

`r` has a finite CI — it is pinned down by the exponential-phase data. `K` does
not: its profile is flat across the prior, so the data place no bound on the
carrying capacity. Extend the window through saturation (as in
`logistic_growth_pipeline.jl`) and `K`'s CI
becomes finite.
""")
end

# ╔═╡ 00000013-0000-0000-0000-000000000000
md"""
## 5  Prediction envelope fans out

`sampleSMPredictions` draws SM parameter vectors from the profile-likelihood
uncertainty region and evaluates the SM at each. Because `K` is unconstrained,
the draws span a wide range of carrying capacities. The resulting band is
**tight at early times**, where the data pin the trajectory, and **fans out
toward the end of the window**, as curves with different carrying capacities
begin to separate. That widening band *is* the non-identifiability, made
visible.
"""

# ╔═╡ 00000014-0000-0000-0000-000000000000
begin
	rng_sample = Random.MersenneTwister(42)
	samples    = sampleSMPredictions(prob, uq; nSamples = 200, rng = rng_sample)
end

# ╔═╡ 00000015-0000-0000-0000-000000000000
plot(samples)

# ╔═╡ 00000016-0000-0000-0000-000000000000
md"""
## Takeaways

- **Identifiability is a joint property of the data and the model** — what's at
  play here is specifically _practical_ identifiability. The logistic SM is
  fully identifiable on a longer window and only partly identifiable on this one,
  even though the model itself is unchanged.
- **Always inspect the profile, not just the point estimate.** `fitSurrogate`
  reports a `K` value regardless; only the profile likelihood reveals that the
  data barely constrain it, so it shouldn't be interpreted as an estimate of the
  true `K`.
- **The prediction envelope localises the problem.** A band that is tight inside
  the data window but fans out beyond it points directly at the unconstrained
  parameter.

Next: see [`cm_posterior_pipeline.jl`](./cm_posterior_pipeline.jl), which uses
`SmoreFit` to build a posterior on CM parameters from real-world observations.
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000007-0000-0000-0000-000000000000
# ╠═00000008-0000-0000-0000-000000000000
# ╟─00000017-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╠═00000009-0000-0000-0000-000000000000
# ╠═0000000a-0000-0000-0000-000000000000
# ╟─0000000b-0000-0000-0000-000000000000
# ╠═0000000c-0000-0000-0000-000000000000
# ╟─0000000d-0000-0000-0000-000000000000
# ╠═0000000e-0000-0000-0000-000000000000
# ╟─0000000f-0000-0000-0000-000000000000
# ╠═00000010-0000-0000-0000-000000000000
# ╠═00000011-0000-0000-0000-000000000000
# ╟─00000012-0000-0000-0000-000000000000
# ╟─00000013-0000-0000-0000-000000000000
# ╠═00000014-0000-0000-0000-000000000000
# ╠═00000015-0000-0000-0000-000000000000
# ╟─00000016-0000-0000-0000-000000000000
