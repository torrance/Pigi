function degridder!(workunits::AbstractVector{WorkUnit{T}}, grid::AbstractMatrix, degridop) where T
    for workunit in workunits
        subgrid = extractsubgrid(grid, workunit)

        fftshift!(subgrid)
        subgridflat = reinterpret(reshape, Complex{T}, subgrid)
        ifft!(subgridflat, (2, 3))

        map!(subgrid, workunit.Aleft, subgrid, workunit.Aright) do Aleft, subgrid, Aright
            return Aleft * subgrid * adjoint(Aright)
        end
        dft!(workunit, subgrid, degridop)
    end
end

function dft!(workunit::WorkUnit{T}, subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop) where T
    lms = fftfreq(workunit.subgridspec.Nx, 1 / workunit.subgridspec.scaleuv)

    Threads.@threads for i in eachindex(workunit.data)
        uvdatum = workunit.data[i]

        data = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
        for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
            phase = -2π * 1im * (
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