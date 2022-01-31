function gridder(workunit::WorkUnit{T}; makepsf::Bool=false) where T
    subgrid = zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    if makepsf
        diftpsf!(subgrid, workunit)
    else
        dift!(subgrid, workunit)
    end

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)
    return fftshift(subgrid)
end

function dift!(subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, workunit::WorkUnit{T}) where T
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))::Frequencies{T}

    Threads.@threads for idx in CartesianIndices(subgrid)
        lpx, mpx = Tuple(idx)
        l, m = lms[lpx], lms[mpx]

        @simd for uvdatum in workunit.data
            phase = 2π * 1im * (
                (uvdatum.u - workunit.u0) * l +
                (uvdatum.v - workunit.v0) * m +
                (uvdatum.w - workunit.w0) * ndash(l, m)
            )
            subgrid[lpx, mpx] += uvdatum.weights .* uvdatum.data * exp(phase)
        end
        subgrid[lpx, mpx] = (
            workunit.Aleft[lpx, mpx] * subgrid[lpx, mpx] * adjoint(workunit.Aright[lpx, mpx])
        )
    end
end

function diftpsf!(subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, workunit::WorkUnit{T}) where T
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))::Frequencies{T}

    Threads.@threads for idx in CartesianIndices(subgrid)
        lpx, mpx = Tuple(idx)
        l, m = lms[lpx], lms[mpx]

        @simd for uvdatum in workunit.data
            phase = 2π * 1im * (
                (uvdatum.u - workunit.u0) * l +
                (uvdatum.v - workunit.v0) * m +
                (uvdatum.w - workunit.w0) * ndash(l, m)
            )
            subgrid[lpx, mpx] += uvdatum.weights * exp(phase)
        end
        subgrid[lpx, mpx] = (
            workunit.Aleft[lpx, mpx] * subgrid[lpx, mpx] * adjoint(workunit.Aright[lpx, mpx])
        )
    end
end