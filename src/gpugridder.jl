function gpugridder(workunit::WorkUnit{T}; makepsf::Bool=false) where T
    subgrid = CUDA.zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    # TODO: Switch memory layout of workunit.data to StructArrays to avoid this fussin' and a feudin'.
    us = CuArray([x.u for x in workunit.data])
    vs = CuArray([x.v for x in workunit.data])
    ws = CuArray([x.w for x in workunit.data])
    weights = CuArray([x.weights for x in workunit.data])
    data = CuArray([x.data for x in workunit.data])

    if makepsf
        kernel = @cuda launch=false gpudiftpsf!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, us, vs, ws, weights, data, workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(workunit.data), config.threads)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, us, vs, ws, weights, data, workunit.subgridspec;
            threads=(nthreads, 1, 1),
            blocks=(1, workunit.subgridspec.Nx, workunit.subgridspec.Ny),
            shmem=sizeof(SMatrix{2, 2, Complex{T}, 4}) * nthreads
        )
    else
        kernel = @cuda launch=false gpudift!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, us, vs, ws, weights, data, workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(workunit.data), config.threads)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, us, vs, ws, weights, data, workunit.subgridspec;
            threads=(nthreads, 1, 1),
            blocks=(1, workunit.subgridspec.Nx, workunit.subgridspec.Ny),
            shmem=sizeof(SMatrix{2, 2, Complex{T}, 4}) * nthreads
        )
    end

    subgrid = Array(subgrid)

    for i in eachindex(workunit.Aleft, subgrid, workunit.Aright)
        subgrid[i] = workunit.Aleft[i] * subgrid[i] * adjoint(workunit.Aright[i])
    end

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)

    return fftshift(subgrid)
end

function gpudift!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, us, vs, ws, weights, data, subgridspec) where T
    shm = CUDA.CuDynamicSharedArray(SMatrix{2, 2, Complex{T}, 4}, blockDim().x)

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}
    lpx, mpx = blockIdx().y, blockIdx().z
    l, m = lms[lpx], lms[mpx]

    cell = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
    for i in threadIdx().x:blockDim().x:length(data)
        phase = 2π * 1im * (
            (us[i] - u0) * l +
            (vs[i] - v0) * m +
            (ws[i] - w0) * ndash(l, m)
        )
        cell += weights[i] .* data[i] * exp(phase)
    end
    shm[threadIdx().x] = cell
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
        subgrid[lpx, mpx] = shm[1]
    end

    return nothing
end

function gpudiftpsf!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, us, vs, ws, weights, data, subgridspec) where T
    shm = CUDA.CuDynamicSharedArray(SMatrix{2, 2, Complex{T}, 4}, blockDim().x)

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}
    lpx, mpx = blockIdx().y, blockIdx().z
    l, m = lms[lpx], lms[mpx]

    cell = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
    for i in threadIdx().x:blockDim().x:length(data)
        phase = 2π * 1im * (
            (us[i] - u0) * l +
            (vs[i] - v0) * m +
            (ws[i] - w0) * ndash(l, m)
        )
        cell += weights[i] * exp(phase)
    end
    shm[threadIdx().x] = cell
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
        subgrid[lpx, mpx] = shm[1]
    end

    return nothing
end