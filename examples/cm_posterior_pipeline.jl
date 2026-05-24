### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000000
# Launch via SmoreExamples.run_example("cm_posterior_pipeline.jl"), or manually:
#
#   using Pluto
#   Pluto.run(notebook                   = "/path/to/cm_posterior_pipeline.jl",
#             workspace_custom_startup_expr = "import Pkg; Pkg.activate(\"/path/to/SmoreExamples\"); Pkg.instantiate()")

# ╔═╡ 00000002-0000-0000-0000-000000000000
begin
	using Smore
	using CairoMakie
end

# ╔═╡ 00000003-0000-0000-0000-000000000000
md"""
# CM Posterior from Real Data (SmoreFit)

This notebook is **Step 8** of the SmoreVerse pipeline. The earlier
`logistic_growth_pipeline.jl` covered:

- Steps 1–6 (`SmoreBase`): define an SM, fit it to CM data, profile likelihood UQ.
- Step 7 (`SmoreGSA`): global sensitivity of SM outputs to CM parameters.

The piece still missing is the *inverse* problem: given **real-world data** and
the SM you already fit per cohort, *which CM parameter sets are consistent with
that data?* That is what `SmoreFit` answers.

## The trick

You already have a profile likelihood per cohort point — the SM-parameter
confidence region induced by the CM running at each cohort point. SmoreFit
profiles the same SM against the **real** observations to get a *data-side*
SM-parameter confidence region in the same space. A cohort point is in the
posterior iff its SM region overlaps the data's. Both sides live in
SM-parameter space, so we can compare them directly — no MCMC, no likelihood
evaluation of the CM, no expensive forward simulations.

This notebook walks through:

1. Build a CM cohort + **real** profile-likelihood UQ at every cohort point.
2. Generate synthetic observational data at an interior CM point.
3. `buildPosterior(...)` and inspect the result.
4. Visualize the posterior on the CM grid.
5. Compare the three bridge methods.
6. Switch to the graded posterior.
7. Query interior CM points off the cohort grid.
"""

# ╔═╡ 00000004-0000-0000-0000-000000000000
md"""
## 1  Surrogate Model

The same exponential decay SM used in Step 7 of `logistic_growth_pipeline.jl`:

$$y(t) = a \, e^{-b\,t}$$

with SM parameters $a$ (amplitude) and $b$ (decay rate). The CM has its own
parameter pair $(\text{cm\_a}, \text{cm\_b})$ that drives the CM's data
generation; in this toy setup the CM parameters map directly to the SM
parameters $(a, b)$, which keeps the ground truth recoverable and lets you
check the result against intuition.
"""

# ╔═╡ 00000005-0000-0000-0000-000000000000
sm = AnalyticalSurrogateModel(
	fn = (t, p, _c) -> begin
		a, b = p
		reshape(a .* exp.(-b .* t), :, 1)
	end,
)

# ╔═╡ 00000006-0000-0000-0000-000000000000
t = collect(range(0.0, 5.0, 12))

# ╔═╡ 00000007-0000-0000-0000-000000000000
md"""
## 2  Cohort UQ — the upstream input

The CM cohort is a 6 × 6 Cartesian grid over $(\text{cm\_a}, \text{cm\_b})$.
At each cohort point we:

1. Run the CM (here: simulate by evaluating the SM at the CM-true parameters).
2. Fit the SM to the resulting CMData → `SMFitResult`.
3. Profile-likelihood UQ → `ProfileLikelihoodResult`.

Unlike the synthetic flat profiles used in Step 7 (which were appropriate for
demonstrating the GSA *machinery* without committing to CM execution),
SmoreFit's bridges need realistic profile shapes — the CI widths, not just the
midpoints, drive the answer. So this notebook does the full fit + UQ at every
cohort point. With 25 points and a closed-form SM it still takes only a
moment.

The data-side noise level (`σ = 0.2`) is large enough that the resulting
SM-parameter CIs span the gaps between cohort points — without that, the
"cohort sampling density" caveat below would trigger and most cohort points
would score zero. In a real workflow, cohort density should be chosen against
the SM-parameter CI widths the CM noise produces.
"""

# ╔═╡ 00000008-0000-0000-0000-000000000000
begin
	cm_a_vals = collect(range(1.0, 5.0, 6))       # 6 levels for cm_a
	cm_b_vals = collect(range(0.2, 0.8, 6))       # 6 levels for cm_b

	# Row-major Cartesian product (cm_a outer, cm_b inner) — matches GridCMSample's
	# expectation that rows form a regular grid.
	cm_as     = repeat(cm_a_vals; inner = length(cm_b_vals))
	cm_bs     = repeat(cm_b_vals; outer = length(cm_a_vals))
	cm_params = [cm_as cm_bs]
	cm_sample = GridCMSample(cm_params)
	cm_prior  = ParameterPrior([1.0, 0.2], [5.0, 0.8]; names = ["cm_a", "cm_b"])
end

# ╔═╡ 00000009-0000-0000-0000-000000000000
md"""
### SM-side prior, fit, and profile-likelihood settings

The SM prior must encompass the entire CM-side range of $(a, b)$ since the SM
will be fit at every cohort point. We give it some slack on both sides.
"""

# ╔═╡ 0000000a-0000-0000-0000-000000000000
begin
	sm_prior = ParameterPrior([0.5, 0.1], [6.0, 1.0]; names = ["a", "b"])
	P0       = [3.0 0.5]
	noise_σ  = 0.2
	plopts   = ProfileLikelihood(n_points = 15, confidence_level = 0.95)
end

# ╔═╡ 0000000b-0000-0000-0000-000000000000
md"""
### Build the cohort

Each cohort point: evaluate the CM (here, the SM at the CM-true $(a, b)$),
wrap as `CMData`, fit, and profile. The resulting `uq_results` vector is
**row-aligned** with `cm_sample.params` — `uq_results[i]` is the profile at
`cm_sample.params[i, :]`. This is the same upstream convention SmoreGSA uses.
"""

# ╔═╡ 0000000c-0000-0000-0000-000000000000
function _cohortUQ(cm_a, cm_b)
	# CM "run": evaluate the SM at the CM-true parameters and add Gaussian noise structure.
	μ_cm = vec(SmoreBase._evaluate(sm, t, [cm_a, cm_b], "default"))
	d_cm = CMData(μ = μ_cm, σ = fill(noise_σ, length(t)), times = t)
	prob = SMFitProblem(sm, d_cm, sm_prior)
	fit  = fitSurrogate(prob, P0)
	return SmoreBase._uq(prob, fit, plopts)
end

# ╔═╡ 0000000d-0000-0000-0000-000000000000
uq_results = [_cohortUQ(cm_params[i, 1], cm_params[i, 2]) for i in 1:size(cm_params, 1)]

# ╔═╡ 0000000e-0000-0000-0000-000000000000
md"""
## 3  Real Observational Data

In a real workflow this is whatever measurement system produced your data.
For this tutorial we generate it at a known CM point that is **between**
cohort grid points — `(cm_a*, cm_b*) = (2.7, 0.45)` — so we can visualize how
the cohort points around it score and check that the highest-scoring cohort
points really are the closest. The data-side noise level matches the cohort
(`σ = 0.2`).
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

`buildPosterior` accepts either an `SMFitProblem` (when you already have one
in hand for the real data) or the `(sm, data, …)` triple — they are
equivalent. The defaults are `bridge = :box_overlap`, `posterior = :accept`,
`acceptance_tol = 0.0`.
"""

# ╔═╡ 00000011-0000-0000-0000-000000000000
post = buildPosterior(sm, data, uq_results, cm_sample, cm_prior)

# ╔═╡ 00000012-0000-0000-0000-000000000000
md"""
### The result

`CMPosteriorResult` holds: the per-cohort `scores` (in `[0,1]`), an `accepted`
`BitVector`, the bridge + posterior choice, the `data_profiles` (SM profile
against the real data), the cohort `uq_results`, and an interpolator over the
CM grid for interior queries.

For a `GridCMSample` cohort, `acceptedGrid` and `scoreGrid` reshape the result
back onto the grid axes — handy for heatmaps.
"""

# ╔═╡ 00000013-0000-0000-0000-000000000000
let
	n_acc = count(post.accepted)
	top_idx = argmax(post.scores)
	Markdown.parse("""
**Summary**

- Cohort size: $(length(post.scores)) points
- Accepted: $(n_acc) ($(round(100 * n_acc / length(post.scores); digits=1))%)
- Top-scoring cohort: index $(top_idx) → `cm_a = $(cm_params[top_idx, 1])`, `cm_b = $(cm_params[top_idx, 2])` (score = $(round(post.scores[top_idx]; digits=4)))
- True data-generating CM point: `(cm_a*, cm_b*) = ($(cm_a_true), $(cm_b_true))`
""")
end

# ╔═╡ 00000014-0000-0000-0000-000000000000
md"""
### Equivalent `SMFitProblem` form

If you already have the data wrapped in an `SMFitProblem` (e.g., to share the
loss with another part of the pipeline), pass it directly:
"""

# ╔═╡ 00000015-0000-0000-0000-000000000000
let
	problem = SMFitProblem(sm, data, sm_prior)
	post_p  = buildPosterior(problem, uq_results, cm_sample, cm_prior)
	(scores_match = post_p.scores ≈ post.scores,
	 accepted_match = post_p.accepted == post.accepted)
end

# ╔═╡ 00000016-0000-0000-0000-000000000000
md"""
## 5  Visualize the Posterior on the CM Grid

`scoreGrid` reshapes `scores` to size `length.(cm_sample.axes)` so it plots
directly as a heatmap. Overlay markers for accepted cohort points and the
true data-generating point.
"""

# ╔═╡ 00000017-0000-0000-0000-000000000000
let
	S = scoreGrid(post)
	A = acceptedGrid(post)

	fig = Figure(size = (640, 460))
	ax  = Axis(fig[1, 1],
		xlabel = "cm_a", ylabel = "cm_b",
		title  = "Posterior score on the CM grid (:box_overlap)",
	)
	hm = heatmap!(ax, cm_a_vals, cm_b_vals, S; colormap = :viridis)
	Colorbar(fig[1, 2], hm; label = "score ∈ [0, 1]")

	# Accepted cohort points (white circles).
	acc_idx = findall(post.accepted)
	if !isempty(acc_idx)
		scatter!(ax,
			cm_params[acc_idx, 1], cm_params[acc_idx, 2];
			color = :white, strokecolor = :black, strokewidth = 1.0, markersize = 12,
			label = "accepted",
		)
	end

	# All cohort points (small grey dots) to show grid layout.
	scatter!(ax,
		cm_params[:, 1], cm_params[:, 2];
		color = (:black, 0.25), markersize = 4,
	)

	# True data-generating CM point (red star).
	scatter!(ax, [cm_a_true], [cm_b_true];
		marker = :star5, color = :red, strokecolor = :black, strokewidth = 1.0,
		markersize = 18, label = "data truth",
	)

	axislegend(ax; position = :rt)
	fig
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
information. Below: all three bridges side by side on the same cohort.
"""

# ╔═╡ 00000019-0000-0000-0000-000000000000
let
	bridges = (:box_overlap, :data_trace_in_box, :symmetric_trace)
	posts   = [
		buildPosterior(sm, data, uq_results, cm_sample, cm_prior; bridge = b)
		for b in bridges
	]

	fig = Figure(size = (1080, 380))
	for (j, (b, pj)) in enumerate(zip(bridges, posts))
		ax = Axis(fig[1, j],
			xlabel = "cm_a", ylabel = "cm_b",
			title  = "bridge = :$b   (n_acc = $(count(pj.accepted)))",
		)
		hm = heatmap!(ax, cm_a_vals, cm_b_vals, scoreGrid(pj);
			colormap = :viridis, colorrange = (0.0, 1.0),
		)
		scatter!(ax, cm_params[:, 1], cm_params[:, 2]; color = (:black, 0.25), markersize = 4)
		acc = findall(pj.accepted)
		if !isempty(acc)
			scatter!(ax, cm_params[acc, 1], cm_params[acc, 2];
				color = :white, strokecolor = :black, strokewidth = 1.0, markersize = 10,
			)
		end
		scatter!(ax, [cm_a_true], [cm_b_true];
			marker = :star5, color = :red, strokecolor = :black, strokewidth = 1.0,
			markersize = 16,
		)
		j == 3 && Colorbar(fig[1, 4], hm; label = "score")
	end
	fig
end

# ╔═╡ 0000001a-0000-0000-0000-000000000000
md"""
### Reading the comparison

All three bridges agree on the high-score region near the true CM point — that
robustness is the value-add of having three options agree. They disagree on
the *shape* of the falloff: `:box_overlap` penalizes mismatched CI widths
geometrically, while `:data_trace_in_box` is asymmetric (it asks only whether
the data sits inside the cohort box, regardless of how big that box is). If
two bridges disagree about whether a cohort point should be accepted, that
point is *boundary* in the SM-parameter space sense — worth flagging for
follow-up rather than trusting either answer.
"""

# ╔═╡ 0000001b-0000-0000-0000-000000000000
md"""
## 7  Graded Posterior

`posterior = :graded` keeps the per-cohort scores as a continuous weight
rather than thresholding to accept/reject. `posteriorWeights` normalizes them
to a discrete distribution over cohort points — useful when you want to keep
the relative confidence information for downstream uses (importance-weighted
expectations, etc.).
"""

# ╔═╡ 0000001c-0000-0000-0000-000000000000
post_graded = buildPosterior(sm, data, uq_results, cm_sample, cm_prior;
	posterior = :graded,
)

# ╔═╡ 0000001d-0000-0000-0000-000000000000
let
	w  = posteriorWeights(post_graded)
	W  = reshapeToGrid(post_graded.cm_sample, w)
	fig = Figure(size = (640, 460))
	ax  = Axis(fig[1, 1],
		xlabel = "cm_a", ylabel = "cm_b",
		title  = "Graded posterior weights  (Σ = $(round(sum(w); digits=4)))",
	)
	hm = heatmap!(ax, cm_a_vals, cm_b_vals, W; colormap = :viridis)
	Colorbar(fig[1, 2], hm; label = "weight")
	scatter!(ax, cm_params[:, 1], cm_params[:, 2]; color = (:black, 0.25), markersize = 4)
	scatter!(ax, [cm_a_true], [cm_b_true];
		marker = :star5, color = :red, strokecolor = :black, strokewidth = 1.0, markersize = 18,
	)
	fig
end

# ╔═╡ 0000001e-0000-0000-0000-000000000000
md"""
## 8  Interior Queries

The cohort is sparse. To evaluate the posterior at a CM point not on the
cohort grid — for visualization, refinement, or downstream sampling —
`posteriorScore(post, θ_cm)` and `inPosterior(post, θ_cm)` interpolate the
per-cohort SM-parameter CI bounds across the CM grid and re-run the bridge
against the data-side profile at the interpolated box. No new SM fits are
done.

Below: evaluate `posteriorScore` on a fine grid in CM space (40 × 40 ≈
1600 queries) and visualize.

> **Caveat.** Linear bound interpolation is only reliable when consecutive
> cohort CIs overlap each other in SM-parameter space. If the CIs are tight
> relative to how much the bound surface moves between cohort points, interior
> queries can drop to zero near boundaries even close to a cohort point that
> *is* in the posterior. The fix is to add cohort points where the geometry
> changes fast — see the SmoreFit README "cohort sampling density" note.
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

	fig = Figure(size = (700, 480))
	ax  = Axis(fig[1, 1],
		xlabel = "cm_a", ylabel = "cm_b",
		title  = "Interior posterior score (40 × 40 fine grid)",
	)
	hm = heatmap!(ax, a_fine, b_fine, S_fine;
		colormap = :viridis, colorrange = (0.0, 1.0),
	)
	Colorbar(fig[1, 2], hm; label = "score")
	scatter!(ax, cm_params[:, 1], cm_params[:, 2];
		color = :white, strokecolor = :black, strokewidth = 0.5, markersize = 5,
		label = "cohort",
	)
	scatter!(ax, [cm_a_true], [cm_b_true];
		marker = :star5, color = :red, strokecolor = :black, strokewidth = 1.0,
		markersize = 18, label = "data truth",
	)
	axislegend(ax; position = :rt)
	fig
end

# ╔═╡ 00000020-0000-0000-0000-000000000000
md"""
### Querying a single point

The vector form returns a scalar; the matrix form (`[k × n_cm_params]`)
returns a vector of scores. `inPosterior` adds a threshold step (default
`post.acceptance_tol`) and returns a `Bool` / `BitVector`.
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

The default threshold is `post.acceptance_tol` (0.0 here — any non-zero score
counts as accepted). Pass `tol` to tighten or loosen on a per-query basis,
without rebuilding the posterior.
"""

# ╔═╡ 00000023-0000-0000-0000-000000000000
let
	θ = [cm_a_true, cm_b_true]
	(
		score_at_truth = posteriorScore(post, θ),
		default_tol    = inPosterior(post, θ),
		tol_0p5        = inPosterior(post, θ; tol = 0.5),
		tol_0p9        = inPosterior(post, θ; tol = 0.9),   # impossibly tight
	)
end

# ╔═╡ 00000024-0000-0000-0000-000000000000
md"""
## 9  Wrap-Up

The full Smore pipeline now reads:

| Step | Sub-package | Function |
|------|-------------|----------|
| 1–4  | SmoreBase | `SMFitProblem`, `fitSurrogate` |
| 5    | SmoreBase | `SmoreBase._uq` (profile likelihood) |
| 6    | SmoreBase | `sampleSMPredictions` |
| 7    | SmoreGSA  | `runSensitivity` |
| **8** | **SmoreFit** | **`buildPosterior`, `posteriorScore`, `inPosterior`** |

The shared backbone is the per-cohort `ProfileLikelihoodResult`. Once that's
in hand:

- **SmoreGSA** treats those profiles as the *uncertainty envelope* and asks
  how SM output varies as CM parameters change.
- **SmoreFit** treats those profiles as one side of a comparison and asks
  which CM parameter values are consistent with real-world data.

Same upstream cost, two complementary downstream answers.

---

### Try it yourself

- Move `(cm_a_true, cm_b_true)` to a corner of CM space — how does the
  posterior shape change? What happens when the truth sits outside the cohort
  grid entirely?
- Reduce the cohort grid to 3 × 3 — does the interior-query interpolation
  still work? Where does it break?
- Change `noise_σ` on the data side only (leave the cohort noise alone) —
  what happens to the size of the posterior?
- Try `bridge = :data_trace_in_box` with `posterior = :graded` and compare
  the weight distribution to `:box_overlap`.
"""

# ╔═╡ Cell order:
# ╟─00000001-0000-0000-0000-000000000000
# ╟─00000002-0000-0000-0000-000000000000
# ╟─00000003-0000-0000-0000-000000000000
# ╟─00000004-0000-0000-0000-000000000000
# ╠═00000005-0000-0000-0000-000000000000
# ╠═00000006-0000-0000-0000-000000000000
# ╟─00000007-0000-0000-0000-000000000000
# ╠═00000008-0000-0000-0000-000000000000
# ╟─00000009-0000-0000-0000-000000000000
# ╠═0000000a-0000-0000-0000-000000000000
# ╟─0000000b-0000-0000-0000-000000000000
# ╠═0000000c-0000-0000-0000-000000000000
# ╠═0000000d-0000-0000-0000-000000000000
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
# ╟─0000001a-0000-0000-0000-000000000000
# ╟─0000001b-0000-0000-0000-000000000000
# ╠═0000001c-0000-0000-0000-000000000000
# ╠═0000001d-0000-0000-0000-000000000000
# ╟─0000001e-0000-0000-0000-000000000000
# ╠═0000001f-0000-0000-0000-000000000000
# ╟─00000020-0000-0000-0000-000000000000
# ╠═00000021-0000-0000-0000-000000000000
# ╟─00000022-0000-0000-0000-000000000000
# ╠═00000023-0000-0000-0000-000000000000
# ╟─00000024-0000-0000-0000-000000000000
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
