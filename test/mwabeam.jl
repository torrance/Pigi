@testset "MWA Beam" begin
    beam = Pigi.MWABeam(zeros(Int, 16))
    azel = map(_ -> (2π * rand(), π/2 * rand()), CartesianIndices((64, 64)))
    azel[1] = (0., π/2)

    A = Pigi.getresponse(beam, azel, 150e6)

    @test size(A) == size(azel)
    @test all(isapprox.(A[1], [
        0.9702102620742318 + 0.24226441621882988im -0.00024691334309719337 - 0.00011722463144868201im
        -0.0002505324468245751 - 0.00012354386674326775im 0.9704127699647864 + 0.24145197429151424im
    ], rtol=1e-4))
end

@testset "MWA beam response from measurement set" begin
    path = "../../testdata/1215555160/1215555160.ms"

    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=96)
    uvdata = collect(mset)

    time = Pigi.meantime(mset, uvdata)
    freq = Pigi.meanfreq(mset, uvdata)

    mwaframe = Pigi.MWAFrame(time)
    mwabeam = Pigi.MWABeam(mset.delays)

    subgridspec = Pigi.GridSpec(128, 128, scalelm=sin(deg2rad(0.6)))
    gridspec = Pigi.GridSpec(1000, 1000, scaleuv=subgridspec.scaleuv)

    coords_radec = Pigi.grid_to_radec(subgridspec, Tuple(mset.phasecenter))
    coords_azel = Pigi.radec_to_azel(mwaframe, coords_radec)

    beamgrid = Pigi.getresponse(mwabeam, coords_azel, freq)

    power = map(beamgrid) do A
        real(sum((A * A')[[1, 4]])) / 2
    end

    powerlarge = Pigi.resample(power, subgridspec, gridspec)

    @test (real(powerlarge[501, 501]) / power[65, 65]) ≈ 1
end