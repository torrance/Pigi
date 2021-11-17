@testset "UVDatum instantiation" begin
    uvdatum = Pigi.UVDatum{Float32}(
        1, 1, 0, 0, 0, @SMatrix([0 0; 0 0]), @SMatrix([0 0; 0 0])
    )
    @test typeof(uvdatum) == Pigi.UVDatum{Float32}

    uvdatum = Pigi.UVDatum{Float64}(
        1, 1, 0, 0, 0, @SMatrix([0 0; 0 0]), @SMatrix([0 0; 0 0])
    )
    @test typeof(uvdatum) == Pigi.UVDatum{Float64}

    @test_throws TypeError Pigi.UVDatum{Int}(
        1, 1, 0, 0, 0, @SMatrix([0 0; 0 0]), @SMatrix([0 0; 0 0])
    )

    @test_throws TypeError Pigi.UVDatum{ComplexF32}(
        1, 1, 0, 0, 0, @SMatrix([0 0; 0 0]), @SMatrix([0 0; 0 0])
    )
end