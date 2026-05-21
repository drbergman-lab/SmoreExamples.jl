using SmoreExamples
using Test

@testset "SmoreExamples.jl" begin
    @test isdir(examplesdir())
    @test isfile(examplepath("logistic_growth_pipeline.jl"))
end
