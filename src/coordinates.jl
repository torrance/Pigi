struct AltAzFrame
    obj::PyObject
end

function lm_to_radec(l, m, origin)
    ra0, dec0 = origin
    n = sqrt(1 - l^2 - m^2)
    ra = ra0 + atan(l, n * cos(dec0) - m * sin(dec0))
    dec = asin(m * cos(dec0) + n * sin(dec0))
    return ra, dec
end

function radec_to_altaz(frame::AltAzFrame, coords::AbstractArray{NTuple{2, Float64}})
    ras, decs = map(first, coords), map(last, coords)

    AstropyCoords = PyCall.pyimport("astropy.coordinates")
    coord = AstropyCoords.SkyCoord(ras, decs, unit=("rad", "rad"))
    altaz = coord.transform_to(frame.obj)

    res = map((alt, az) -> (deg2rad(alt), deg2rad(az)), altaz.alt, altaz.az)
    return reshape(res, size(coords))
end

function grid_to_radec(gridspec::GridSpec, origin::NTuple{2, Float64})
    return map(CartesianIndices((gridspec.Nx, gridspec.Ny))) do lmpx
        lpx, mpx = Tuple(lmpx)
        l, m = px2sky(lpx, mpx, gridspec)
        ra, dec = lm_to_radec(l, m, origin)
        return (ra, dec)
    end
end