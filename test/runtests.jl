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

    @testset "_example_title" begin
        # H1 inside the first md block is used as the title
        with = """
        # ╔═╡ 00000001-0000-0000-0000-000000000000
        md\"\"\"
        # My Title

        body text
        \"\"\"
        """
        # No H1 anywhere: must fall back to the file name, NOT a `# ╔═╡` cell marker
        without = """
        # ╔═╡ 00000001-0000-0000-0000-000000000000
        md\"\"\"
        just some text, no heading
        \"\"\"

        # ╔═╡ 00000002-0000-0000-0000-000000000000
        x = 1
        """
        mktempdir() do d
            p1 = joinpath(d, "with_h1.jl");   write(p1, with)
            p2 = joinpath(d, "no_h1.jl");      write(p2, without)
            @test SmoreExamples._example_title(p1) == "My Title"
            @test SmoreExamples._example_title(p2) == "no_h1"
        end
    end
end
