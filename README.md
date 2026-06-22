# SmoreExamples

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://drbergman-lab.github.io/SmoreExamples.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://drbergman-lab.github.io/SmoreExamples.jl/dev/)
[![Build Status](https://github.com/drbergman-lab/SmoreExamples.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/drbergman-lab/SmoreExamples.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/drbergman-lab/SmoreExamples.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/drbergman-lab/SmoreExamples.jl)

## Bundled notebooks

| File | What it covers |
|------|---------------|
| `logistic_growth_pipeline.jl` | End-to-end tour of `SmoreBase` (fit + profile-likelihood UQ + prediction sampling) and `SmoreGSA` (EFAST / Morris global sensitivity). |
| `cm_posterior_pipeline.jl` | Step 8 of the pipeline using `SmoreFit`: build a posterior on CM parameters from real-world data, compare bridge methods, query interior CM points. |
| `single_obs_custom_data.jl` | Define a custom `AbstractCMData` subtype for a single-observation + constant-CV noise model. |
| `replicate_runs_custom_loss.jl` | Custom `AbstractCMData` that retains raw replicate runs, plus a custom MAE loss for robust fitting. |

## Running a notebook

Loading the package in an interactive session prints the available examples. To
suppress that, set the environment variable before loading:

```julia
ENV["SMOREEXAMPLES_QUIET"] = "true"
using SmoreExamples
```

You can also see the list at any time by calling `list_examples`:

```julia
using SmoreExamples
list_examples()
```

The easiest way to open one is `run_example`:

```julia
run_example("cm_posterior_pipeline.jl")
```

That copies the bundled notebooks to a temp directory (pass `dir = "/your/path"`
to keep edits) and opens the named one in Pluto with the `SmoreExamples` project
activated. Omit the name to open `logistic_growth_pipeline.jl`.

If you prefer to run Pluto yourself:

```julia
using Pluto
Pluto.run()
```

…and open any of the files above from the Pluto file picker. The notebook
activates the `SmoreExamples` project and instantiates dependencies on startup.
