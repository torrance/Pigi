function degridder!(workunits::AbstractVector{WorkUnit{T}}, grid::AbstractMatrix, subtaper::Matrix{T}, degridop) where T
    for workunit in workunits
        subgrid = extractsubgrid(grid, workunit)

        fftshift!(subgrid)
        ifft!(subgrid)

        map!(subgrid, workunit.Aleft, subgrid, workunit.Aright, subtaper) do Aleft, subgrid, Aright, t
            return Aleft * subgrid * Aright' * t
        end
        dft!(workunit, subgrid, degridop)
    end
end

function dft!(workunit::WorkUnit{T}, subgrid::Matrix{LinearData{T}}, degridop) where T
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))

    Threads.@threads for i in eachindex(workunit.data)
        uvdatum = workunit.data[i]

        data = zero(LinearData{T})
        for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
            phase = -2im * T(π) * (
                (uvdatum.u - workunit.u0) * l +
                (uvdatum.v - workunit.v0) * m +
                (uvdatum.w - workunit.w0) * ndash(l, m)
            )
            data += subgrid[lpx, mpx] * exp(phase)
        end

        workunit.data.data[i] = degridop(uvdatum.data, data)
    end
end

@inline function degridop_replace(_, new)
    return new
end

@inline function degridop_subtract(old, new)
    return old - new
end

@inline function degridop_add(old, new)
    return old + new
end