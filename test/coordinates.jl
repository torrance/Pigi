@testset "RA/Dec to Alt/Az conversion" begin
    coords_radec = map(range(1, 100)) do _
        (rand() * 2π, (rand() - 0.5) * π)
    end

    mjd = 58312.92595655624
    frame = Pigi.MWAFrame(mjd)

    coords_azel = Pigi.radec_to_azel(frame, coords_radec)

    # Compare to Astropy coordinates
    expected = let pos=frame[2], frame
        AstropyCoords = PyCall.pyimport("astropy.coordinates")
        AstropyTime = PyCall.pyimport("astropy.time")

        # Construct frame
        obstime = AstropyTime.Time(mjd, format="mjd")
        location = AstropyCoords.EarthLocation.from_geodetic(
            ustrip(u"°", Pigi.long(pos)), ustrip(u"°", Pigi.lat(pos));
            height=ustrip(u"m", Pigi.radius(pos))
        )
        frame = AstropyCoords.AltAz(;location, obstime)

        # Perform the conversion
        ras, decs = map(first, coords_radec), map(last, coords_radec)
        coords = AstropyCoords.SkyCoord(ras, decs, unit=("rad", "rad"))
        azel = coords.transform_to(frame)
        res = map((az, alt) -> (deg2rad(az), deg2rad(alt)), azel.az, azel.alt)
        reshape(res, size(coords_radec))
    end

    @test all(zip(expected, coords_azel)) do (a, b)
        b = ustrip.(Float64, u"rad", b)
        isapprox(mod(a[1], 2π), mod(b[1], 2π), atol=5e-2) && isapprox(a[2], b[2], atol=5e-2)
    end
end