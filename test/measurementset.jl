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
end