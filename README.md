# SmoreExamples

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://drbergman-lab.github.io/SmoreExamples.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://drbergman-lab.github.io/SmoreExamples.jl/dev/)
[![Build Status](https://github.com/drbergman-lab/SmoreExamples.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/drbergman-lab/SmoreExamples.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/drbergman-lab/SmoreExamples.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/drbergman-lab/SmoreExamples.jl)

## Running the Pluto example

Start Pluto from Julia with:

```julia
using Pluto
Pluto.run()
```

Then open `examples/logistic_growth_pipeline.jl` from the Pluto file picker.

If you installed this package and want Julia to tell you where that example lives,
you can do:

```julia
using SmoreExamples
SmoreExamples.examplepath("logistic_growth_pipeline.jl")
```

That returns the full path to the bundled example file, which you can paste into
Pluto's "Open a notebook" dialog.

The notebook activates the `SmoreExamples` project and instantiates dependencies
on startup, so opening the file is usually all you need.
