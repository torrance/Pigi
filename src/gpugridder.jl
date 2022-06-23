function gridder!(grid::CuMatrix, workunits::AbstractVector{WorkUnit{T}}, subtaper::CuMatrix{T}; makepsf::Bool=false) where T
    subgridspec = workunits[1].subgridspec
    subgrids = CuArray{SMatrix{2, 2, Complex{T}, 4}, 3}(undef, subgridspec.Nx, subgridspec.Ny, length(workunits))

    # A terms are shared amongst work units, so we only have to transfer over the unique arrays.
    # We do this (awkwardly) by hashing them into an ID dictionary.
    Aterms = IdDict{AbstractMatrix{SMatrix{2, 2, Complex{T}, 4}}, CuMatrix{SMatrix{2, 2, Complex{T}, 4}}}()
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
            gpudift!(subgrid, origin, uvdata, subgridspec, makepsf)

            # Apply A terms (and normalise prior to fft)
            Aleft = Aterms[workunit.Aleft]
            Aright = Aterms[workunit.Aright]
            map!(subgrid, Aleft, subgrid, Aright, subtaper) do Aleft, subgrid, Aright, t
                return Aleft * subgrid * adjoint(Aright) * t / (subgridspec.Nx * subgridspec.Ny)
            end
        end
    end

    # Perform fft in the default stream, to avoid expensive internal allocations by fft.
    # Additionally, addsubgrid must be applied sequentially.
    for (i, workunit) in enumerate(workunits)
        subgrid = view(subgrids, :, :, i)

        # Apply fft
        subgridflat = reinterpret(reshape, Complex{T}, subgrid)
        fft!(subgridflat, (2, 3))
        fftshift!(subgrid)

        addsubgrid!(grid, subgrid, workunit)
    end
end

function gpudift!(subgrid::CuMatrix{SMatrix{2, 2, Complex{T}, 4}}, origin, uvdata, subgridspec, makepsf::Bool) where {T}
    function _gpudift!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, origin, uvdata, subgridspec, ::Val{makepsf}) where {T, makepsf}
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}

        if idx > length(subgrid)
            return nothing
        end

        lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])
        l, m = lms[lpx], lms[mpx]

        cell = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
        for i in 1:length(uvdata)
            phase = 2π * 1im * (
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

        subgrid[idx] = cell
        return nothing
    end

    kernel = @cuda launch=false _gpudift!(
        subgrid, origin, uvdata, subgridspec, Val(makepsf)
    )
    config = launch_configuration(kernel.fun)
    threads = min(subgridspec.Nx * subgridspec.Ny, config.threads)
    blocks = cld(subgridspec.Nx * subgridspec.Ny, threads)
    kernel(
        subgrid, origin, uvdata, subgridspec, Val(makepsf);
        threads, blocks
    )
end