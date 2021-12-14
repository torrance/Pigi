function degridder!(subgrid::Subgrid, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}) where T
    grid = ifftshift(grid)
    gridflat = reinterpret(reshape, Complex{T}, grid)
    ifft!(gridflat, (2, 3))

    for i in eachindex(subgrid.Aleft, grid, subgrid.Aright)
        grid[i] = subgrid.Aleft[i] * grid[i] * adjoint(subgrid.Aright[i])
    end
    dft!(subgrid, grid)
end

function dft!(subgrid::Subgrid{T}, grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}) where T
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
            data,
        )
    end
end