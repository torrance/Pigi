@testset "MWAFrame" begin
    frame = Pigi.MWAFrame(5039601370.0 / (24 * 3600))
    alt, az = Pigi.radec_to_altaz(frame, [(0., 0.)])[1]
    @test alt ≈ 0.7903214637847765
    @test az ≈ 1.0346009564373215

    coords = map(_ -> (rand() - 0.5, rand() - 0.5), CartesianIndices((100, 100)))
    altazs = Pigi.radec_to_altaz(frame, coords)
    @test size(altazs) == size(coords)
end

@testset "MWA Beam" begin
    beam = Pigi.MWABeam(zeros(Int, 16))
    altaz = map(_ -> (π/2 * rand(), 2π * rand()), CartesianIndices((64, 64)))
    altaz[1] = (π/2, 0.)

    A = Pigi.getresponse(beam, altaz, 150e6)

    @test size(A) == size(altaz)
    @test all(isapprox.(A[1], [
        9.70412770e-01 + 2.41451974e-01im -2.50532447e-04 - 1.23543867e-04im; -2.46913343e-04 - 1.17224631e-04im 9.70210262e-01 + 2.42264416e-01im
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
    coords_altaz = Pigi.radec_to_altaz(mwaframe, coords_radec)

    beamgrid = Pigi.getresponse(mwabeam, coords_altaz, freq)

    power = map(beamgrid) do A
        real(sum((A * A')[[1, 4]])) / 2
    end

    powerlarge = Pigi.resample(power, subgridspec, gridspec)

    @test (real(powerlarge[501, 501]) / power[65, 65]) ≈ 1
end