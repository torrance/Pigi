function degridder!(workunit::WorkUnit, subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, degridop, ::Type{CuArray}) where T
    subgrid = ifftshift(subgrid)
    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    ifft!(subgridflat, (2, 3))

    for i in eachindex(workunit.Aleft, subgrid, workunit.Aright)
        subgrid[i] = workunit.Aleft[i] * subgrid[i] * adjoint(workunit.Aright[i])
    end

    uvdata = replace_storage(CuArray, workunit.data)
    subgridd = CuArray(subgrid)

    kernel = @cuda launch=false gpudft!(uvdata, workunit.u0, workunit.v0, workunit.w0, subgridd, workunit.subgridspec, degridop)
    config = launch_configuration(kernel.fun)
    nthreads = min(length(uvdata), config.threads)
    nblocks = cld(length(uvdata), nthreads)
    kernel(uvdata, workunit.u0, workunit.v0, workunit.w0, subgridd, workunit.subgridspec, degridop; threads=nthreads, blocks=nblocks)

    copyto!(workunit.data.data, uvdata.data)
end

function gpudft!(uvdata::StructVector{Pigi.UVDatum{T}}, u0, v0, w0, subgrid, subgridspec, degridop) where T
    gpuidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    lms = fftfreq(subgridspec.Nx, 1 / subgridspec.scaleuv)

    for idx in gpuidx:stride:length(uvdata)
        uvdatum = uvdata[idx]

        data = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
        for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
            phase = -2π * 1im * (
                (uvdatum.u - u0) * l +
                (uvdatum.v - v0) * m +
                (uvdatum.w - w0) * ndash(l, m)
            )
            data += subgrid[lpx, mpx] * exp(phase)
        end

        uvdata.data[idx] = degridop(uvdatum.data, data)
    end
end