function degridder!(subgrid::Subgrid, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop) where T
    grid = ifftshift(grid)
    gridflat = reinterpret(reshape, Complex{T}, grid)
    ifft!(gridflat, (2, 3))

    for i in eachindex(subgrid.Aleft, grid, subgrid.Aright)
        grid[i] = subgrid.Aleft[i] * grid[i] * adjoint(subgrid.Aright[i])
    end
    dft!(subgrid, grid, degridop)
end

function dft!(subgrid::Subgrid{T}, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop) where T
    lms = fftfreq(subgrid.subgridspec.Nx, 1 / subgrid.subgridspec.scaleuv)

    Threads.@threads for i in eachindex(subgrid.data)
        uvdatum = subgrid.data[i]

        data = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
        for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
            phase = -2π * 1im * (
                (uvdatum.u - subgrid.u0) * l +
                (uvdatum.v - subgrid.v0) * m +
                (uvdatum.w - subgrid.w0) * ndash(l, m)
            )
            data += grid[lpx, mpx] * exp(phase)
        end

        subgrid.data[i] = UVDatum{T}(
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