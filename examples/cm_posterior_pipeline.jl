### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 00000002-0000-0000-0000-000000000000
begin
	using Smore
	using Plots
end

# ╔═╡ 00000001-0000-0000-0000-000000000000
# Launch via SmoreExamples.run_example("cm_posterior_pipeline.jl"), or manually:
#
#   using Pluto
#   Pluto.run(notebook                   = "/path/to/cm_posterior_pipeline.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# CM Posterior from Real Data (SmoreFit)

This notebook is **Step 8** of the SmoreVerse pipeline. The earlier
`logistic_growth_pipeline.jl` covered:

- Steps 1–6 (`SmoreBase`): define an SM, fit it to CM data, profile likelihood UQ.
- Step 7 (`SmoreGSA`): global sensitivity of SM outputs to CM parameters.

The piece still missing is the *inverse* problem: given **real-world data** and
the SM you already fit per CM parameter set, *which CM parameter sets are consistent with
that data?* That is what `SmoreFit` answers.

## The trick

You already have a profile likelihood per CM parameter set — the SM-parameter
confidence region induced by the CM running at each one. SmoreFit
profiles the same SM against the **real** observations to get a *data-side*
SM-parameter confidence region in the same space. A CM parameter set is in the
posterior iff its SM region overlaps the data's. Both sides live in
SM-parameter space, so we can compare them directly — no MCMC, no likelihood
evaluation of the CM, no expensive forward simulations.

This notebook walks through:

1. Build a set of CM parameter sets + **real** profile-likelihood UQ at every one.
2. Generate synthetic observational data at an interior CM point.
3. `buildPosterior(...)` and inspect the result.
4. Visualize the posterior on the CM grid.
5. Compare the three bridge methods.
6. Read the graded posterior from the same result.
7. Query interior CM points off the CM parameter set grid.
"""

# ╔═╡ 00000007-0000-0000-0000-000000000000
md"""
## 1  CM Parameter Set UQ — the upstream input

The CM parameter sets form a 6 × 6 Cartesian grid over $(\text{cm\_a}, \text{cm\_b})$.
At each one we:

1. Run the CM (here: simulate by evaluating the SM — described in Section 2 below — at the
   CM-true parameters).
2. Fit the SM to the resulting CMData → `SMFitResult`.
3. Profile-likelihood UQ → `ProfileLikelihoodResult`.

Unlike the synthetic flat profiles used in Step 7 (which were appropriate for
demonstrating the GSA *machinery* without committing to CM execution),
SmoreFit's bridges need realistic profile shapes — the CI widths, not just the
midpoints, drive the answer. So this notebook does the full fit + UQ at every
CM parameter set. With 25 points and a closed-form SM it still takes only a
moment.

The data-side noise level (`σ = 0.2`) is large enough that the resulting
SM-parameter CIs span the gaps between CM parameter sets — without that, the
"CM parameter set sampling density" caveat below would trigger and most CM
parameter sets would score zero. In a real workflow, CM parameter set density
should be chosen against the SM-parameter CI widths the CM noise produces.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
# The SM, described properly in Section 2 below; needed here to generate the CM parameter sets'
# data (a simple exponential decay function chosen for its closed-form CM → SM mapping).
sm = CustomSurrogateModel(
	fn = (t, p, _c) -> begin
		a, b = p
		reshape(a .* exp.(-b .* t), :, 1)
	end,
)

# ╔═╡ 00000006-0000-0000-0000-000000000000
t = collect(range(0.0, 5.0, 12))

# ╔═╡ 00000008-0000-0000-0000-000000000000
begin
	cm_a_vals = collect(range(1.0, 5.0, 6))       # 6 levels for cm_a
	cm_b_vals = collect(range(0.2, 0.8, 6))       # 6 levels for cm_b

	cm_sample = GridCMSample(cm_a_vals, cm_b_vals; names = ["cm_a", "cm_b"])
	cm_params = cm_sample.params
end

# ╔═╡ 00000009-0000-0000-0000-000000000000
md"""
### SM-side prior, fit, and profile-likelihood settings

The SM prior is the one used for fitting at every CM parameter set, so it must
be wide enough to cover the region the fitted SM parameters could plausibly
land in — not just the fit's expected neighborhood.
"""

# ╔═╡ 0000000a-0000-0000-0000-000000000000
begin
	sm_prior = ParameterPrior([0.5, 0.1], [6.0, 1.0]; names = ["a", "b"])
	P0       = [3.0, 0.5]   # single shared guess, broadcast to every CM parameter set below
	noise_σ  = 0.2
	pl_opts  = ProfileLikelihood(n_points = 15, confidence_level = 0.95)
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
md"""
### Build the CM parameter sets

One `CMData` holds all 36 CM parameter sets at once (its `cm_param_sets`
axis), so fitting and profiling all of them is a single `fitSurrogate` call
and a single `quantifyUncertainty` call. The resulting
`Vector{ProfileLikelihoodResult}` is **row-aligned** with `cm_sample.params`
(`uq_results[i]` is the profile at `cm_sample.params[i, :]`) — the same
upstream convention SmoreGSA uses.
"""

# ╔═╡ 0000000c-0000-0000-0000-000000000000
begin
	# CM "run" for every CM parameter set: evaluate the SM at each one's CM-true parameters.
	n_ps = size(cm_params, 1)
	μ_cm_param_sets = reduce(hcat, [vec(SmoreBase._evaluate(sm, t, cm_params[i, :], "default")) for i in 1:n_ps])
	d_cm_param_sets = CMData(μ = μ_cm_param_sets, σ = fill(noise_σ, size(μ_cm_param_sets)), times = t, cm_param_sets = n_ps)
	prob_cm_param_sets = SMFitProblem(sm, d_cm_param_sets, sm_prior)
	fit_cm_param_sets  = fitSurrogate(prob_cm_param_sets, P0)   # vector P0 broadcasts the shared guess to every row
end

# ╔═╡ 0000000d-0000-0000-0000-000000000000
uq_results = quantifyUncertainty(pl_opts, prob_cm_param_sets, fit_cm_param_sets)

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 2  The Surrogate Model

Having used it above to generate the CM parameter sets' data, here's the SM itself: a simple
exponential decay function, chosen for its closed-form CM → SM mapping:

$$y(t) = a \, e^{-b\,t}$$

with SM parameters $a$ (amplitude) and $b$ (decay rate). The CM has its own parameter pair
$(\text{cm\_a}, \text{cm\_b})$ that drives its data generation; in this toy setup the CM
parameters map directly to the SM parameters $(a, b)$, which keeps the ground truth recoverable
and lets you check the result against intuition.
"""

# ╔═╡ 0000000e-0000-0000-0000-000000000000
md"""
## 3  Real Observational Data

In a real workflow, this is whatever measurement system produced your data — a real-world
experiment, not a CM run. For this tutorial we stand in for that experiment by running the CM at
a single hidden parameter point and adding measurement noise, exactly as Section 1 did for each
CM parameter set — the difference is that this point, `(cm_a*, cm_b*) = (2.7, 0.45)`, is treated
as unknown to the rest of the pipeline; recovering (a neighborhood of) it is the point of
`buildPosterior`. We place it **between** CM parameter set grid points so we can visualize how
the CM parameter sets around it score and check that the highest-scoring ones really are the
closest. The data-side noise level matches the CM parameter sets (`σ = 0.2`).
"""

# ╔═╡ 0000000f-0000-0000-0000-000000000000
begin
	cm_a_true = 2.7
	cm_b_true = 0.45
	μ_data    = vec(SmoreBase._evaluate(sm, t, [cm_a_true, cm_b_true], "default"))
	data      = CMData(μ = μ_data, σ = fill(noise_σ, length(t)), times = t)
end

# ╔═╡ 00000010-0000-0000-0000-000000000000
md"""
## 4  Build the Posterior
"""

# ╔═╡ 00000011-0000-0000-0000-000000000000
post = buildPosterior(sm, data, uq_results, cm_sample)

# ╔═╡ 00000012-0000-0000-0000-000000000000
md"""
### The result

`CMPosteriorResult` holds: the per-CM-parameter-set `scores` (in `[0,1]`), an `accepted`
`BitVector`, the bridge choice, the `data_profiles` (SM profile
against the real data), the CM parameter sets' `uq_results`, and an interpolator over the
CM grid for interior queries.

For a `GridCMSample` of CM parameter sets, `acceptedGrid` and `scoreGrid` reshape the result
back onto the grid axes — handy for heatmaps.
"""

# ╔═╡ 00000013-0000-0000-0000-000000000000
let
	n_acc = count(post.accepted)
	top_idx = argmax(post.scores)
	Markdown.parse("""
**Summary**

- CM parameter set count: $(length(post.scores)) points
- Accepted: $(n_acc) ($(round(100 * n_acc / length(post.scores); digits=1))%)
- Top-scoring CM parameter set: index $(top_idx) → `cm_a = $(cm_params[top_idx, 1])`, `cm_b = $(cm_params[top_idx, 2])` (score = $(round(post.scores[top_idx]; digits=4)))
- True data-generating CM point: `(cm_a*, cm_b*) = ($(cm_a_true), $(cm_b_true))`
""")
end

# ╔═╡ 00000014-0000-0000-0000-000000000000
md"""
### Equivalent `SMFitProblem` form

`buildPosterior` also accepts an `SMFitProblem` directly instead of the
`(sm, data, …)` triple — handy if you already have one in hand (e.g., to
share the loss with another part of the pipeline). The two forms are
equivalent; defaults are `bridge = :box_overlap`, `min_score = 0.0`.
"""

# ╔═╡ 00000015-0000-0000-0000-000000000000
let
	problem = SMFitProblem(sm, data, sm_prior)
	post_p  = buildPosterior(problem, uq_results, cm_sample)
	(scores_match = post_p.scores ≈ post.scores,
	 accepted_match = post_p.accepted == post.accepted)
end

# ╔═╡ 00000016-0000-0000-0000-000000000000
md"""
## 5  Visualize the Posterior on the CM Grid

`scoreGrid` reshapes `scores` to size `length.(cm_sample.axes)` so it plots
directly as a heatmap. Overlay markers for accepted CM parameter sets and the
true data-generating point.
"""

# ╔═╡ 00000017-0000-0000-0000-000000000000
let
	S = scoreGrid(post)

	# heatmap(x, y, Z) in Plots indexes Z[row = y, col = x], so transpose scoreGrid.
	plt = heatmap(cm_a_vals, cm_b_vals, S';
		c = :viridis, colorbar_title = "score ∈ [0, 1]", size = (640, 460),
		xlabel = "cm_a", ylabel = "cm_b",
		title = "Posterior score on the CM grid (:box_overlap)", legend = :topright)

	# All CM param_sets (small grey dots) to show grid layout.
	scatter!(plt, cm_params[:, 1], cm_params[:, 2];
		markercolor = :black, markeralpha = 0.25, markersize = 3, label = "")

	# Accepted CM param_sets (white circles).
	acc_idx = findall(post.accepted)
	if !isempty(acc_idx)
		scatter!(plt, cm_params[acc_idx, 1], cm_params[acc_idx, 2];
			markercolor = :white, markerstrokecolor = :black, markerstrokewidth = 1,
			markersize = 6, label = "accepted")
	end

	# True data-generating CM point (red star).
	scatter!(plt, [cm_a_true], [cm_b_true];
		marker = :star5, markercolor = :red, markerstrokecolor = :black,
		markersize = 9, label = "data truth")
	plt
end

# ╔═╡ 00000018-0000-0000-0000-000000000000
md"""
## 6  Bridge-Method Comparison

The bridge picks how the two SM-parameter confidence regions are compared:

| Bridge | Score |
|---|---|
| `:box_overlap` | Relative overlap volume of the two marginal-CI hyper-rectangles |
| `:data_trace_in_box` | Fraction of data profile trace points inside the CM box |
| `:symmetric_trace` | Max of the two-way trace-in-box fractions |

`:box_overlap` is symmetric and cheap; the trace methods give more nuanced
overlap on regions where the profile shape (not just its CI edges) carries
information. Below: all three bridges side by side on the same CM parameter sets.
"""

# ╔═╡ 00000019-0000-0000-0000-000000000000
let
	bridges = (:box_overlap, :data_trace_in_box, :symmetric_trace)
	posts   = [
		buildPosterior(sm, data, uq_results, cm_sample; bridge = b)
		for b in bridges
	]

	panels = map(zip(bridges, posts)) do (b, pj)
		p = heatmap(cm_a_vals, cm_b_vals, scoreGrid(pj)';
			c = :viridis, clims = (0.0, 1.0), colorbar_title = "score",
			xlabel = "cm_a", ylabel = "cm_b", legend = false,
			title = "bridge = :$b   (n_acc = $(count(pj.accepted)))")
		scatter!(p, cm_params[:, 1], cm_params[:, 2];
			markercolor = :black, markeralpha = 0.25, markersize = 3, label = "")
		acc = findall(pj.accepted)
		isempty(acc) || scatter!(p, cm_params[acc, 1], cm_params[acc, 2];
			markercolor = :white, markerstrokecolor = :black, markerstrokewidth = 1,
			markersize = 5, label = "")
		scatter!(p, [cm_a_true], [cm_b_true];
			marker = :star5, markercolor = :red, markerstrokecolor = :black,
			markersize = 8, label = "")
		p
	end
	plot(panels...; layout = (1, 3), size = (1080, 380))
end

# ╔═╡ 0000001b-0000-0000-0000-000000000000
md"""
## 7  Graded Posterior

`accepted`/`posteriorSamples` (used above) threshold the scores to a hard accept/reject set.
`posteriorWeights` instead normalizes the same `scores` — already stored on `post`, no
new `buildPosterior` call needed — to a discrete distribution over CM parameter sets,
keeping the relative confidence information for downstream uses (importance-weighted
expectations, etc.).
"""

# ╔═╡ 0000001d-0000-0000-0000-000000000000
let
	w  = posteriorWeights(post)
	W  = reshapeToGrid(post.cm_sample, w)
	plt = heatmap(cm_a_vals, cm_b_vals, W';
		c = :viridis, colorbar_title = "weight", size = (640, 460),
		xlabel = "cm_a", ylabel = "cm_b", legend = false,
		title = "Graded posterior weights  (Σ = $(round(sum(w); digits=4)))")
	scatter!(plt, cm_params[:, 1], cm_params[:, 2];
		markercolor = :black, markeralpha = 0.25, markersize = 3, label = "")
	scatter!(plt, [cm_a_true], [cm_b_true];
		marker = :star5, markercolor = :red, markerstrokecolor = :black, markersize = 9, label = "")
	plt
end

# ╔═╡ 0000001e-0000-0000-0000-000000000000
md"""
## 8  Interior Queries

The CM parameter sets are sparse. To evaluate the posterior at a CM point not among
them — for visualization, refinement, or downstream sampling —
`posteriorScore(post, θ_cm)` and `inPosterior(post, θ_cm)` interpolate the
per-CM-parameter-set SM-parameter CI bounds across the CM grid and re-run the bridge
against the data-side profile at the interpolated box. No new SM fits are
done.

Below: evaluate `posteriorScore` on a fine grid in CM space (40 × 40 ≈
1600 queries) and visualize.

> **Caveat.** Linear bound interpolation is best when consecutive
> CM parameter sets' CIs overlap each other in SM-parameter space. If the CIs are tight
> relative to how much the bound surface moves between CM parameter sets, interior
> queries can be unreliable even close to a CM parameter set that
> *is* in the posterior. The fix is to add CM parameter sets where the geometry
> changes fast — see the SmoreFit README "CM param_set sampling density" note.
"""

# ╔═╡ 0000001f-0000-0000-0000-000000000000
let
	# Fine grid of CM queries.
	a_fine = collect(range(cm_a_vals[1], cm_a_vals[end], 40))
	b_fine = collect(range(cm_b_vals[1], cm_b_vals[end], 40))

	queries = [reshape([a, b], 1, 2) for a in a_fine, b in b_fine]
	queries = reduce(vcat, queries)         # [(40*40) × 2]
	scores_fine = posteriorScore(post, queries)
	S_fine = reshape(scores_fine, length(a_fine), length(b_fine))

	plt = heatmap(a_fine, b_fine, S_fine';
		c = :viridis, clims = (0.0, 1.0), colorbar_title = "score", size = (700, 480),
		xlabel = "cm_a", ylabel = "cm_b", legend = :topright,
		title = "Interior posterior score (40 × 40 fine grid)")
	scatter!(plt, cm_params[:, 1], cm_params[:, 2];
		markercolor = :white, markerstrokecolor = :black, markerstrokewidth = 0.5,
		markersize = 3, label = "CM parameter sets")
	scatter!(plt, [cm_a_true], [cm_b_true];
		marker = :star5, markercolor = :red, markerstrokecolor = :black,
		markersize = 9, label = "data truth")
	plt
end

# ╔═╡ 00000020-0000-0000-0000-000000000000
md"""
### Querying a single point

The vector form returns a scalar; the matrix form (`[k × n_cm_params]`)
returns a vector of scores. `inPosterior` adds a threshold step (default
`post.min_score`) and returns a `Bool` / `BitVector`.
"""

# ╔═╡ 00000021-0000-0000-0000-000000000000
(
	score_at_truth = posteriorScore(post, [cm_a_true, cm_b_true]),
	accepted_at_truth = inPosterior(post, [cm_a_true, cm_b_true]),
	score_far  = posteriorScore(post, [1.2, 0.75]),
	accepted_far = inPosterior(post, [1.2, 0.75]),
)

# ╔═╡ 00000022-0000-0000-0000-000000000000
md"""
### Threshold override

The default threshold is `post.min_score` (0.0 here — any non-zero score
counts as accepted). Pass `min_score` to tighten or loosen on a per-query basis,
without rebuilding the posterior.
"""

# ╔═╡ 00000023-0000-0000-0000-000000000000
let
	θ = [cm_a_true, cm_b_true]
	(
		score_at_truth    = posteriorScore(post, θ),
		default_min_score = inPosterior(post, θ),
		min_score_0p5     = inPosterior(post, θ; min_score = 0.5),
		min_score_0p9     = inPosterior(post, θ; min_score = 0.9),   # impossibly tight
	)
end

# ╔═╡ 00000024-0000-0000-0000-000000000000
md"""
## 9  Wrap-Up

The full Smore pipeline now reads:

| Step | Sub-package | Function |
|------|-------------|----------|
| 1–4  | SmoreBase | `SMFitProblem`, `fitSurrogate` |
| 5    | SmoreBase | `quantifyUncertainty` (profile likelihood) |
| 6    | SmoreBase | `sampleSMPredictions` |
| 7    | SmoreGSA  | `runSensitivity` |
| **8** | **SmoreFit** | **`buildPosterior`, `posteriorScore`, `inPosterior`** |

The shared backbone is the per-CM-parameter-set `ProfileLikelihoodResult`. Once that's
in hand:

- **SmoreGSA** treats those profiles as the *uncertainty envelope* and asks
  how SM output varies as CM parameters change.
- **SmoreFit** treats those profiles as one side of a comparison and asks
  which CM parameter values are consistent with real-world data.

Same upstream cost, two complementary downstream answers.

---

### Try it yourself

- Move `(cm_a_true, cm_b_true)` to a corner of CM space — how does the
  posterior shape change? What happens when the truth sits outside the CM parameter set
  grid entirely?
- Reduce the CM parameter set grid to 3 × 3 — does the interior-query interpolation
  still work? Where does it break?
- Change `noise_σ` on the data side only (leave the CM parameter sets' noise alone) —
  what happens to the size of the posterior?
- Try `bridge = :data_trace_in_box` and compare `posteriorWeights`' distribution
  to `:box_overlap`'s.
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─00000007-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╠═00000008-0000-0000-0000-000000000000
# ╟─00000009-0000-0000-0000-000000000000
# ╠═0000000a-0000-0000-0000-000000000000
# ╟─0000000b-0000-0000-0000-000000000000
# ╠═0000000c-0000-0000-0000-000000000000
# ╠═0000000d-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╟─0000000e-0000-0000-0000-000000000000
# ╠═0000000f-0000-0000-0000-000000000000
# ╟─00000010-0000-0000-0000-000000000000
# ╠═00000011-0000-0000-0000-000000000000
# ╟─00000012-0000-0000-0000-000000000000
# ╠═00000013-0000-0000-0000-000000000000
# ╟─00000014-0000-0000-0000-000000000000
# ╠═00000015-0000-0000-0000-000000000000
# ╟─00000016-0000-0000-0000-000000000000
# ╠═00000017-0000-0000-0000-000000000000
# ╟─00000018-0000-0000-0000-000000000000
# ╠═00000019-0000-0000-0000-000000000000
# ╟─0000001b-0000-0000-0000-000000000000
# ╠═0000001d-0000-0000-0000-000000000000
# ╟─0000001e-0000-0000-0000-000000000000
# ╠═0000001f-0000-0000-0000-000000000000
# ╟─00000020-0000-0000-0000-000000000000
# ╠═00000021-0000-0000-0000-000000000000
# ╟─00000022-0000-0000-0000-000000000000
# ╠═00000023-0000-0000-0000-000000000000
# ╟─00000024-0000-0000-0000-000000000000
