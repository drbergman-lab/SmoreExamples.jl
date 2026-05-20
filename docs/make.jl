using SMoReExamples
using Documenter

DocMeta.setdocmeta!(SMoReExamples, :DocTestSetup, :(using SMoReExamples); recursive=true)

makedocs(;
    modules=[SMoReExamples],
    authors="Daniel Bergman <danielrbergman@gmail.com> and contributors",
    sitename="SMoReExamples.jl",
    format=Documenter.HTML(;
        canonical="https://drbergman-lab.github.io/SMoReExamples.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/drbergman-lab/SMoReExamples.jl",
    devbranch="main",
)
