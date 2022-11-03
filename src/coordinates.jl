function lm_to_radec(l, m, origin)
    ra0, dec0 = origin
    n = sqrt(1 - l^2 - m^2)
    ra = ra0 + atan(l, n * cos(dec0) - m * sin(dec0))
    dec = asin(m * cos(dec0) + n * sin(dec0))
    return ra, dec
end

function radec_to_azel(frame, coords::AbstractArray{NTuple{2, Float64}})
    direction = zero(Measures.Direction)
    c = Measures.Converter(Measures.Directions.J2000, Measures.Directions.AZEL, frame...)

    return map(coords) do (ra, dec)
        direction.type = Measures.Directions.J2000
        direction.long = ra
        direction.lat = dec
        mconvert!(direction, direction, c)
        return ustrip(Float64, u"rad", direction.long), ustrip(Float64, u"rad", direction.lat)
    end
end

function grid_to_radec(gridspec::GridSpec, origin::NTuple{2, Float64})
    return map(CartesianIndices((gridspec.Nx, gridspec.Ny))) do lmpx
        lpx, mpx = Tuple(lmpx)
        l, m = px2sky(lpx, mpx, gridspec)
        ra, dec = lm_to_radec(l, m, origin)
        return (ra, dec)
    end
end