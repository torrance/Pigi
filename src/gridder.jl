function gridder(subgrid::Subgrid{T}; makepsf::Bool=false) where T
    grid = zeros(
        SMatrix{2, 2, Complex{T}, 4}, subgrid.subgridspec.Nx, subgrid.subgridspec.Ny
    )

    if makepsf
        diftpsf!(grid, subgrid)
    else
        dift!(grid, subgrid)
    end

    gridflat = reinterpret(reshape, Complex{T}, grid)
    fft!(gridflat, (2, 3))
    grid ./= (subgrid.subgridspec.Nx * subgrid.subgridspec.Ny)
    return fftshift(grid)
end

function dift!(grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, subgrid::Subgrid{T}) where T
    lms = fftfreq(subgrid.subgridspec.Nx, T(1 / subgrid.subgridspec.scaleuv))::Frequencies{T}

    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        @simd for uvdatum in subgrid.data
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

function diftpsf!(grid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, subgrid::Subgrid{T}) where T
    lms = fftfreq(subgrid.subgridspec.Nx, T(1 / subgrid.subgridspec.scaleuv))::Frequencies{T}

    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        @simd for uvdatum in subgrid.data
            phase = 2π * 1im * (
                (uvdatum.u - subgrid.u0) * l +
                (uvdatum.v - subgrid.v0) * m +
                (uvdatum.w - subgrid.w0) * ndash(l, m)
            )
            grid[lpx, mpx] += uvdatum.weights * exp(phase)
        end
        grid[lpx, mpx] = (
            subgrid.Aleft[lpx, mpx] * grid[lpx, mpx] * adjoint(subgrid.Aright[lpx, mpx])
        )
    end
end