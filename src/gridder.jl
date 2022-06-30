function gridder!(grid::AbstractMatrix, workunits::AbstractVector{WorkUnit{T}}, subtaper::Matrix{T}; makepsf::Bool=false) where T
    subgridspec = workunits[1].subgridspec
    subgrid = Matrix{SMatrix{2, 2, Complex{T}, 4}}(undef, subgridspec.Nx, subgridspec.Ny)

    for workunit in workunits
        fill!(subgrid, zero(SMatrix{2, 2, Complex{T}, 4}))
        dift!(subgrid, workunit, Val(makepsf))

        # Apply taper and normalise prior to fft. Also apply Aterms if we are not making a PSF.
        if makepsf
            map!(subgrid, subgrid, subtaper) do subgrid, t
                return subgrid * t / (subgridspec.Nx * subgridspec.Ny)
            end
        else
            map!(subgrid, workunit.Aleft, subgrid, workunit.Aright, subtaper) do Aleft, subgrid, Aright, t
                return inv(Aleft) * subgrid * adjoint(inv(Aright)) * t / (subgridspec.Nx * subgridspec.Ny)
            end
        end

        subgridflat = reinterpret(reshape, Complex{T}, subgrid)
        fft!(subgridflat, (2, 3))
        fftshift!(subgrid)

        addsubgrid!(grid, subgrid, workunit)
    end
end

function dift!(subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, workunit::WorkUnit{T}, ::Val{makepsf}) where {T, makepsf}
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))::Frequencies{T}
    uvdata = workunit.data

    Threads.@threads for idx in CartesianIndices(subgrid)
        lpx, mpx = Tuple(idx)
        l, m = lms[lpx], lms[mpx]

        @simd for i in 1:length(uvdata)
            phase = 2π * 1im * (
                (uvdata.u[i] - workunit.u0) * l +
                (uvdata.v[i] - workunit.v0) * m +
                (uvdata.w[i] - workunit.w0) * ndash(l, m)
            )
            if makepsf
                subgrid[lpx, mpx] += uvdata.weights[i] * exp(phase)
            else
                subgrid[lpx, mpx] += uvdata.weights[i] .* uvdata.data[i] * exp(phase)
            end
        end
    end
end