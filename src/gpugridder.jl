function gridder(workunit::WorkUnit{T}, ::Type{CuArray}; makepsf::Bool=false) where T
    subgridd = CUDA.zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    uvdata = replace_storage(CuArray, workunit.data)

    kernel = @cuda launch=false gpudift!(
        subgridd, workunit.u0, workunit.v0, workunit.w0, uvdata.u, uvdata.v, uvdata.w, uvdata.weights, uvdata.data, workunit.subgridspec, Val(makepsf)
    )
    config = launch_configuration(kernel.fun)
    nthreads = min(length(subgridd), config.threads)
    nblocks = cld(length(subgridd), nthreads)
    kernel(
        subgridd, workunit.u0, workunit.v0, workunit.w0, uvdata.u, uvdata.v, uvdata.w, uvdata.weights, uvdata.data, workunit.subgridspec, Val(makepsf);
        threads=nthreads,
        blocks=nblocks,
    )

    subgrid = Array(subgridd)

    for i in eachindex(workunit.Aleft, subgrid, workunit.Aright)
        subgrid[i] = workunit.Aleft[i] * subgrid[i] * adjoint(workunit.Aright[i])
    end

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)

    fftshift!(subgrid)
    return subgrid
end

function gpudift!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, us, vs, ws, weights, data, subgridspec, ::Val{makepsf}) where {T, makepsf}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}

    if idx > length(subgrid)
        return nothing
    end

    lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])
    l, m = lms[lpx], lms[mpx]

    cell = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
    for i in 1:length(data)
        phase = 2π * 1im * (
            (us[i] - u0) * l +
            (vs[i] - v0) * m +
            (ws[i] - w0) * ndash(l, m)
        )
        if makepsf
            cell += weights[i] * exp(phase)
        else
            cell += weights[i] .* data[i] * exp(phase)
        end
    end

    subgrid[idx] = cell
    return nothing
end