```@meta
CurrentModule = SmoreExamples
```

# SmoreExamples

Documentation for [SmoreExamples](https://github.com/drbergman-lab/SmoreExamples.jl).

A bundle of runnable [Pluto](https://plutojl.org) notebooks demonstrating the
SmoreVerse modeling pipeline.

## Available examples

Call [`list_examples`](@ref) to see every bundled notebook and its title:

```@example
using SmoreExamples # hide
list_examples()
```

Open any of them in Pluto with [`run_example`](@ref):

```julia
using SmoreExamples
run_example("cm_posterior_pipeline.jl")
```

The notebooks are copied to a temporary directory (pass `dir = "/your/path"` to
keep edits) and opened with the `SmoreExamples` project activated. Omit the name
to open `logistic_growth_pipeline.jl`.

## API

```@index
```

```@autodocs
Modules = [SmoreExamples]
```
