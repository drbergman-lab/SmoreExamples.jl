module SmoreExamples

export examplesdir, examplepath

examplesdir() = joinpath(pkgdir(SmoreExamples), "examples")

examplepath(name::AbstractString) = joinpath(examplesdir(), name)

end
