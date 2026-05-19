using SMoReExamples
using Documenter

DocMeta.setdocmeta!(SMoReExamples, :DocTestSetup, :(using SMoReExamples); recursive=true)

makedocs(;
    modules=[SMoReExamples],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SMoReExamples.jl",
    format=Documenter.HTML(;
        canonical="https://Daniel Bergman.github.io/SMoReExamples.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Daniel Bergman/SMoReExamples.jl",
    devbranch="main",
)
