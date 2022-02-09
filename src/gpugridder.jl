function gpugridder(workunit::WorkUnit{T}; makepsf::Bool=false) where T
    subgrid = CUDA.zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    if makepsf
        kernel = @cuda launch=false gpudiftpsf!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(subgrid), config.threads)
        nblocks = cld(length(subgrid), nthreads)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec;
            threads=nthreads, blocks=nblocks
        )
    else
        kernel = @cuda launch=false gpudift!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec
        )
        config = launch_configuration(kernel.fun)
        nthreads = min(length(subgrid), config.threads)
        nblocks = cld(length(subgrid), nthreads)
        kernel(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec;
            threads=nthreads, blocks=nblocks
        )
    end

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)

    return fftshift(Array(subgrid))
end

function gpudift!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, uvdata, Aleft, Aright, subgridspec) where T
    gpuidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}

    for idx in gpuidx:stride:length(subgrid)
        lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])
        l, m = lms[lpx], lms[mpx]

        for uvdatum in uvdata
            phase = 2π * 1im * (
                (uvdatum.u - u0) * l +
                (uvdatum.v - v0) * m +
                (uvdatum.w - w0) * ndash(l, m)
            )
            subgrid[lpx, mpx] += uvdatum.weights .* uvdatum.data * exp(phase)
        end
        subgrid[lpx, mpx] = (
            Aleft[lpx, mpx] * subgrid[lpx, mpx] * adjoint(Aright[lpx, mpx])
        )
    end

    return nothing
end

function gpudiftpsf!(subgrid::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, u0, v0, w0, uvdata, Aleft, Aright, subgridspec) where T
    gpuidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))::Frequencies{T}

    for idx in gpuidx:stride:length(subgrid)
        lpx, mpx = Tuple(CartesianIndices(subgrid)[idx])
        l, m = lms[lpx], lms[mpx]

        for uvdatum in uvdata
            phase = 2π * 1im * (
                (uvdatum.u - u0) * l +
                (uvdatum.v - v0) * m +
                (uvdatum.w - w0) * ndash(l, m)
            )
            subgrid[lpx, mpx] += uvdatum.weights * exp(phase)
        end
        subgrid[lpx, mpx] = (
            Aleft[lpx, mpx] * subgrid[lpx, mpx] * adjoint(Aright[lpx, mpx])
        )
    end

    return nothing
end