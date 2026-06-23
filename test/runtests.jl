using SmoreExamples
using Test

@testset "SmoreExamples.jl" begin
    @test run_example isa Function

    examples = list_examples()
    @test !isempty(examples)
    @test all(endswith(e.file, ".jl") for e in examples)
    @test all(!isempty(e.title) for e in examples)
    # every listed file actually exists in the examples directory
    @test all(isfile(joinpath(SmoreExamples.examplesdir(), e.file)) for e in examples)
end
