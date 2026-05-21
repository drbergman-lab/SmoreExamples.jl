using SmoreExamples
using Documenter

DocMeta.setdocmeta!(SmoreExamples, :DocTestSetup, :(using SmoreExamples); recursive=true)

makedocs(;
    modules=[SmoreExamples],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SmoreExamples.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/SmoreExamples.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/SmoreExamples.jl",
    devbranch="main",
)
