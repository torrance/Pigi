function gridder!(
    grid::AbstractMatrix{S},
    workunits::AbstractVector{WorkUnit{T}},
    subtaper::Matrix{T};
    makepsf::Bool=false
) where {T, S <: OutputType{T}}
    subgridspec = workunits[1].subgridspec
    subgrid = Matrix{S}(undef, subgridspec.Nx, subgridspec.Ny)

    for workunit in workunits
        fill!(subgrid, zero(S))
        dift!(subgrid, workunit, Val(makepsf))

        # Apply taper and normalise prior to fft.
        map!(subgrid, subgrid, subtaper) do subgrid, t
            return subgrid * t / (subgridspec.Nx * subgridspec.Ny)
        end

        fft!(subgrid)
        fftshift!(subgrid)

        addsubgrid!(grid, subgrid, workunit)
    end
end

function dift!(
    subgrid::Matrix{S},
    workunit::WorkUnit{T},
    ::Val{makepsf}
) where {T, S <: OutputType{T}, makepsf}
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))::Frequencies{T}
    uvdata = workunit.data

    Threads.@threads for idx in CartesianIndices(subgrid)
        lpx, mpx = Tuple(idx)
        l, m = lms[lpx], lms[mpx]

        cell = zero(LinearData{T})
        @simd for i in 1:length(uvdata)
            phase = 2im * T(π) * (
                (uvdata.u[i] - workunit.u0) * l +
                (uvdata.v[i] - workunit.v0) * m +
                (uvdata.w[i] - workunit.w0) * ndash(l, m)
            )
            if makepsf
                cell += uvdata.weights[i] * exp(phase)
            else
                cell += uvdata.weights[i] .* uvdata.data[i] * exp(phase)
            end
        end

        if makepsf
            subgrid[idx] = cell
        else
            subgrid[idx] = workunit.Aleft[idx], cell, workunit.Aright[idx]
        end
    end
end