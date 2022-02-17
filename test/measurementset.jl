@testset "Measurement set" begin
    path = "../../testdata/1215555160/1215555160.ms"

    mset = Pigi.MeasurementSet(path)
    @test length(mset.freqs) == 768
    @test length(mset.timesteps) == 30
    @test mset.nrows == 228780

    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=96)
    @test length(mset.freqs) == 96
    @test mset.nrows == 228780
end

@testset "Measurement set reading" begin
    path = "../../testdata/1215555160/1215555160.ms"

    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=192)
    uvdata = [uvdatum for uvdatum in Pigi.read(mset)]

    @test length(uvdata) == mset.nrows * length(mset.freqs)
    @test !all(all(isnan.(u.data)) for u in uvdata)
    @test !all(all(isnan.(u.weights)) for u in uvdata)
    @test allunique((u.row, u.chan) for u in uvdata)
end

@testset "Measurement set precision" begin
    path = "../../testdata/1215555160/1215555160.ms"

    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=92)

    uvdata64 = collect(Pigi.read(mset))
    uvdata32 = collect(Pigi.read(mset, precision=Float32))

    @test typeof(uvdata32[1]) == Pigi.UVDatum{Float32}
    @test typeof(uvdata64[1]) == Pigi.UVDatum{Float64}
    @test all(isapprox(u1.u, u2.u) for (u1, u2) in zip(uvdata64, uvdata32))
    @test all(isapprox(u1.weights, u2.weights) for (u1, u2) in zip(uvdata64, uvdata32))
    @test all(isapprox(u1.data, u2.data) for (u1, u2) in zip(uvdata64, uvdata32))
end