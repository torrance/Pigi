function degridder!(workunit::WorkUnit, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop, ::Type{Array}) where T
    grid = ifftshift(grid)
    gridflat = reinterpret(reshape, Complex{T}, grid)
    ifft!(gridflat, (2, 3))

    for i in eachindex(workunit.Aleft, grid, workunit.Aright)
        grid[i] = workunit.Aleft[i] * grid[i] * adjoint(workunit.Aright[i])
    end
    dft!(workunit, grid, degridop)
end

function dft!(workunit::WorkUnit{T}, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop) where T
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
            data += grid[lpx, mpx] * exp(phase)
        end

        workunit.data[i] = UVDatum{T}(
            uvdatum.row,
            uvdatum.chan,
            uvdatum.u,
            uvdatum.v,
            uvdatum.w,
            uvdatum.weights,
            degridop(uvdatum.data, data),
        )
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