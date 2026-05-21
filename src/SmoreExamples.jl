module SmoreExamples

import Pluto

export run_example

examplesdir() = joinpath(pkgdir(SmoreExamples), "examples")

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

end
