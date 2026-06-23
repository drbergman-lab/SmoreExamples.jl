module SmoreExamples

import Pluto

export run_example, list_examples

examplesdir() = joinpath(pkgdir(SmoreExamples), "examples")

"""
    ExampleInfo

A single bundled example notebook: its `file` name (pass this to
[`run_example`](@ref)) and the `title` taken from the notebook's first heading.
"""
struct ExampleInfo
    file::String
    title::String
end

"""
    ExampleList

Result of [`list_examples`](@ref): an indexable, iterable collection of
[`ExampleInfo`](@ref) that prints as a table.
"""
struct ExampleList <: AbstractVector{ExampleInfo}
    examples::Vector{ExampleInfo}
end

Base.size(l::ExampleList) = size(l.examples)
Base.getindex(l::ExampleList, i::Int) = l.examples[i]

function Base.show(io::IO, ::MIME"text/plain", l::ExampleList)
    if isempty(l)
        print(io, "No example notebooks found in $(examplesdir())")
        return
    end
    width = maximum(length(e.file) for e in l)
    println(io, "$(length(l)) example notebook$(length(l) == 1 ? "" : "s") "
                * "(run with `run_example(\"<file>\")`):")
    for e in l
        println(io, "  ", rpad(e.file, width), "  ", e.title)
    end
    print(io, "\nLaunch one with, e.g., `run_example(\"$(first(l).file)\")`.")
end

# Pull the first Markdown H1 (`# ...`) out of a Pluto notebook's `md\"\"\"` block.
# Only `# ` lines *inside* an md block count, so Pluto cell markers (`# ╔═╡ …`),
# which sit between blocks, are never mistaken for a heading. Falls back to the
# file name (sans extension) if no heading is found.
function _example_title(path::AbstractString)
    in_md = false
    for line in eachline(path)
        s = strip(line)
        if !in_md
            occursin("md\"\"\"", s) && (in_md = true)
        elseif startswith(s, "# ")
            return strip(s[3:end])
        elseif s == "\"\"\""
            in_md = false   # standalone closing delimiter — md block ended without an H1
        end
    end
    return replace(basename(path), r"\.jl$" => "")
end

"""
    list_examples() -> ExampleList

List the example notebooks bundled with SmoreExamples, with the title of each.

Pass any listed `file` to [`run_example`](@ref) to open it in Pluto.

# Example

```julia
julia> list_examples()
```
"""
function list_examples()
    files = sort!(filter(f -> endswith(f, ".jl"), readdir(examplesdir())))
    return ExampleList([ExampleInfo(f, _example_title(joinpath(examplesdir(), f)))
                        for f in files])
end

"""
    run_example(name = "logistic_growth_pipeline.jl"; dir = nothing)

Open the bundled example notebook `name` in Pluto with the `SmoreExamples`
project activated. See [`list_examples`](@ref) for the available `name`s.

The notebooks are copied to `dir` (a temporary directory by default; pass a path
to keep your edits) before being opened.
"""
function run_example(name::AbstractString = "logistic_growth_pipeline.jl";
                     dir::Union{AbstractString, Nothing} = nothing)
    dest_dir = dir === nothing ? mktempdir(; prefix = "SmoreExamples_") : (mkpath(dir); dir)

    for fname in readdir(examplesdir())
        endswith(fname, ".jl") || continue
        dest_file = joinpath(dest_dir, fname)
        isfile(dest_file) && continue
        cp(joinpath(examplesdir(), fname), dest_file)
        chmod(dest_file, 0o644)
    end

    if dir === nothing
        @warn "Examples copied to a temporary directory — any edits will be lost when Julia exits.\n" *
              "Pass `dir = \"/your/path\"` to save to a persistent location." _file=nothing
    end
    dest = joinpath(dest_dir, name)
    println("Opening: $dest")
    startup = "import Pkg; Pkg.activate($(repr(pkgdir(SmoreExamples)))); Pkg.instantiate()"
    Pluto.run(notebook = dest, workspace_custom_startup_expr = startup)
end

function __init__()
    # Greet the user with the available examples when the package loads.
    # Stay quiet during tests, docs builds, and other non-interactive sessions.
    (isinteractive() && get(ENV, "SMOREEXAMPLES_QUIET", "") != "true") || return
    try
        io = stderr
        printstyled(io, "SmoreExamples\n"; bold = true)
        show(io, MIME"text/plain"(), list_examples())
        printstyled(io, "\n\nSet ENV[\"SMOREEXAMPLES_QUIET\"] = \"true\" before "
                      * "`using SmoreExamples` to hide this message.\n"; color = :light_black)
    catch err
        @debug "SmoreExamples.__init__ could not list examples" exception = err
    end
    return
end

end
