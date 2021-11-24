function gridder(subgrid::Subgrid{T}) where T
    grid = zeros(
        SMatrix{2, 2, Complex{T}, 4}, subgrid.subgridspec.Nx, subgrid.subgridspec.Ny
    )
    dift!(grid, subgrid)
    gridflat = reinterpret(reshape, Complex{T}, grid)
    fft!(gridflat, (2, 3))
    grid ./= (subgrid.subgridspec.Nx * subgrid.subgridspec.Ny)
    return fftshift(grid)
end

function dift!(grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, subgrid::Subgrid{T}) where T
    lms = fftfreq(subgrid.subgridspec.Nx, T(1 / subgrid.subgridspec.scaleuv))::Frequencies{T}

    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        for uvdatum in subgrid.data
            phase = 2π * 1im * (
                (uvdatum.u - subgrid.u0) * l +
                (uvdatum.v - subgrid.v0) * m +
                (uvdatum.w - subgrid.w0) * ndash(l, m)
            )
            grid[lpx, mpx] += uvdatum.weights .* uvdatum.data * exp(phase)
        end
        grid[lpx, mpx] = (
            subgrid.Aleft[lpx, mpx] * grid[lpx, mpx] * adjoint(subgrid.Aright[lpx, mpx])
        )
    end
end