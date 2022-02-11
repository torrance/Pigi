function gpugridder(workunit::WorkUnit{T}; makepsf::Bool=false) where T
    subgrid = CUDA.zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    uvdata = CuArray(workunit.data)
    Aleft = CuArray(workunit.Aleft)
    Aright = CuArray(workunit.Aright)

    if makepsf
        kernel = @cuda launch=false gpudiftpsf!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, uvdata, Aleft, Aright, workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(uvdata), config.threads)
        nblocks = length(subgrid)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, uvdata, Aleft, Aright, workunit.subgridspec;
            threads=nthreads, blocks=nblocks, shmem=sizeof(SMatrix{2, 2, Complex{T}, 4}) * nthreads
        )
    else
        kernel = @cuda launch=false gpudift!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, uvdata, Aleft, Aright, workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(uvdata), config.threads)
        nblocks = length(subgrid)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, uvdata, Aleft, Aright, workunit.subgridspec;
            threads=nthreads, blocks=nblocks, shmem=sizeof(SMatrix{2, 2, Complex{T}, 4}) * nthreads
        )
    end

    CUDA.unsafe_free!(uvdata)
    CUDA.unsafe_free!(Aleft)
    CUDA.unsafe_free!(Aright)

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)

    return fftshift(Array(subgrid))
end

function gpudift!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, uvdata, Aleft, Aright, subgridspec) where T
    shm = CUDA.CuDynamicSharedArray(SMatrix{2, 2, Complex{T}, 4}, blockDim().x)
    shm[threadIdx().x] = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}
    lpx, mpx = Tuple(CartesianIndices(subgrid)[blockIdx().x])
    l, m = lms[lpx], lms[mpx]

    for i in threadIdx().x:blockDim().x:length(uvdata)
        uvdatum = uvdata[i]

        phase = 2π * 1im * (
            (uvdatum.u - u0) * l +
            (uvdatum.v - v0) * m +
            (uvdatum.w - w0) * ndash(l, m)
        )
        shm[threadIdx().x] += uvdatum.weights .* uvdatum.data * exp(phase)
    end
    CUDA.sync_threads()

    # Perform sum reduction over shared memory using interleaved addressing
    s = 1
    while s < blockDim().x
        i = 2 * (threadIdx().x - 1) * s + 1
        if i + s <= blockDim().x
            shm[i] += shm[i + s]
        end
        s *= 2
        CUDA.sync_threads()
    end

    if threadIdx().x == 1
        subgrid[lpx, mpx] = Aleft[lpx, mpx] * shm[1] * adjoint(Aright[lpx, mpx])
    end

    return nothing
end

function gpudiftpsf!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, uvdata, Aleft, Aright, subgridspec) where T
    shm = CUDA.CuDynamicSharedArray(SMatrix{2, 2, Complex{T}, 4}, blockDim().x)
    shm[threadIdx().x] = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}
    lpx, mpx = Tuple(CartesianIndices(subgrid)[blockIdx().x])
    l, m = lms[lpx], lms[mpx]

    for i in threadIdx().x:blockDim().x:length(uvdata)
        uvdatum = uvdata[i]

        phase = 2π * 1im * (
            (uvdatum.u - u0) * l +
            (uvdatum.v - v0) * m +
            (uvdatum.w - w0) * ndash(l, m)
        )
        shm[threadIdx().x] += uvdatum.weights * exp(phase)
    end
    CUDA.sync_threads()

    # Perform sum reduction over shared memory using interleaved addressing
    s = 1
    while s < blockDim().x
        i = 2 * (threadIdx().x - 1) * s + 1
        if i + s <= blockDim().x
            shm[i] += shm[i + s]
        end
        s *= 2
        CUDA.sync_threads()
    end

    if threadIdx().x == 1
        subgrid[lpx, mpx] = Aleft[lpx, mpx] * shm[1] * adjoint(Aright[lpx, mpx])
    end

    return nothing
end