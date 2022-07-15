function gridder!(
    grid::CuArray{S},
    workunits::AbstractVector{WorkUnit{T}},
    subtaper::CuMatrix{T};
    makepsf::Bool=false
) where {T, S <: OutputType{T}}
    subgridspec = workunits[1].subgridspec
    subgrids = CuArray{S}(undef, subgridspec.Nx, subgridspec.Ny, length(workunits))

    # A terms are shared amongst work units, so we only have to transfer over the unique arrays.
    # We do this (awkwardly) by hashing them into an ID dictionary.
    Aterms = IdDict{AbstractMatrix{Comp2x2{T}}, CuMatrix{Comp2x2{T}}}()
    for workunit in workunits, Aterm in (workunit.Aleft, workunit.Aright)
        if !haskey(Aterms, Aterm)
            Aterms[Aterm] = CuArray(Aterm)
        end
    end
    CUDA.synchronize()

    Base.@sync for (i, workunit) in enumerate(workunits)
        Base.@async CUDA.@sync begin
            origin = (u0=workunit.u0, v0=workunit.v0, w0=workunit.w0)
            subgrid = view(subgrids, :, :, i)
            uvdata = replace_storage(CuArray, workunit.data)

            # Perform direct IFT
            Aleft, Aright = Aterms[workunit.Aleft], Aterms[workunit.Aright]
            gpudift!(subgrid, Aleft, Aright, origin, uvdata, subgridspec, makepsf)

            # Apply taper and normalise prior to fft. Also apply Aterms if we are not making a PSF.
            map!(subgrid, subgrid, subtaper) do cell, t
                return cell * t / (subgridspec.Nx * subgridspec.Ny)
            end
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
    subgrid::CuMatrix{S}, Aleft, Aright, origin, uvdata, subgridspec, makepsf::Bool
) where {T, S <: OutputType{T}}
    function _gpudift!(
        subgrid::CuDeviceMatrix{S}, Aleft, Aright, origin, uvdata, subgridspec, ::Val{makepsf}
    ) where {T, S <: OutputType{T}, makepsf}
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}

        if idx > length(subgrid)
            return nothing
        end

        lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])
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

        return nothing
    end

    kernel = @cuda launch=false _gpudift!(
        subgrid, Aleft, Aright, origin, uvdata, subgridspec, Val(makepsf)
    )
    config = launch_configuration(kernel.fun)
    threads = min(subgridspec.Nx * subgridspec.Ny, config.threads)
    blocks = cld(subgridspec.Nx * subgridspec.Ny, threads)
    kernel(
        subgrid, Aleft, Aright, origin, uvdata, subgridspec, Val(makepsf);
        threads, blocks
    )
end