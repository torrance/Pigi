function gpugridder(workunit::WorkUnit{T}; makepsf::Bool=false) where T
    subgrid = CUDA.zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    nblocks = ceil(Int, length(workunit.data) / 256)
    if makepsf
        @cuda blocks=nblocks threads=256 gpudiftpsf!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec
        )
    else
        @cuda blocks=nblocks threads=256 gpudift!(
            subgrid, workunit.u0, workunit.v0, workunit.w0, CuArray(workunit.data), CuArray(workunit.Aleft), CuArray(workunit.Aright), workunit.subgridspec
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

    uvdatum = uvdata[1]

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

    uvdatum = uvdata[1]

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