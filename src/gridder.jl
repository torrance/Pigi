function gridder!(
    grid::AbstractMatrix{S},
    workunits::AbstractVector{WorkUnit{T}},
    subtaper::AbstractMatrix{T};
    makepsf::Bool=false
) where {T, S <: OutputType{T}}
    wrapper = getwrapper(grid)
    subgridspec = workunits[1].subgridspec
    subgrids = wrapper{S}(undef, subgridspec.Nx, subgridspec.Ny, length(workunits))

    # A terms are shared amongst work units, so we only have to transfer over the unique arrays.
    # We do this (awkwardly) by hashing them into an ID dictionary.
    Aterms = IdDict{AbstractMatrix{Comp2x2{T}}, wrapper{Comp2x2{T}}}()
    for workunit in workunits, Aterm in (workunit.Aleft, workunit.Aright)
        if !haskey(Aterms, Aterm)
            Aterms[Aterm] = wrapper(Aterm)
        end
    end
    CUDA.synchronize()

    Base.@sync for (i, workunit) in enumerate(workunits)
        Base.@async begin
            origin = (u0=workunit.u0, v0=workunit.v0, w0=workunit.w0)
            subgrid = view(subgrids, :, :, i)
            uvdata = replace_storage(wrapper, workunit.data)

            # Perform direct IFT
            Aleft, Aright = Aterms[workunit.Aleft], Aterms[workunit.Aright]
            gpudift!(subgrid, Aleft, Aright, origin, uvdata, subgridspec, makepsf)

            # Apply taper and normalise prior to fft. Also apply Aterms if we are not making a PSF.
            map!(subgrid, subgrid, subtaper) do cell, t
                return cell * t / (subgridspec.Nx * subgridspec.Ny)
            end

            CUDA.synchronize()
        end
    end

    # Perform fft in the default stream, to avoid expensive internal allocations by fft.
    # Additionally, addsubgrid must be applied sequentially.
    for (i, workunit) in enumerate(workunits)
        subgrid = view(subgrids, :, :, i)

        # Apply fft
        fft!(subgrid)
        fftshift!(subgrid)

        addsubgrid!(grid, subgrid, workunit)
    end
end

function gpudift!(
    subgrid::AbstractMatrix{S}, Aleft, Aright, origin, uvdata, subgridspec, makepsf::Bool
) where {T, S <: OutputType{T}}
    @kernel function _gpudift!(
        subgrid::AbstractMatrix{S}, Aleft, Aright, origin, uvdata, subgridspec, ::Val{makepsf}
    ) where {T, S <: OutputType{T}, makepsf}
        idx = @index(Global)
        lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])

        lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}
        l, m = lms[lpx], lms[mpx]

        cell = zero(LinearData{T})
        for i in 1:length(uvdata)
            phase = 2im * T(π) * (
                (uvdata.u[i] - origin.u0) * l +
                (uvdata.v[i] - origin.v0) * m +
                (uvdata.w[i] - origin.w0) * ndash(l, m)
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
            s::S = LinearData{T}(
                inv(Aleft[idx]) * cell * inv(Aright[idx])'
            )
            subgrid[idx] = normalize(s, Aleft[idx], Aright[idx])
        end
    end

    kernel = _gpudift!(kernelconf(subgrid)...)
    wait(
        kernel(subgrid, Aleft, Aright, origin, uvdata, subgridspec, Val(makepsf); ndrange=length(subgrid))
    )
end