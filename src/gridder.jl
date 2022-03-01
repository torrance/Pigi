function gridder(workunit::WorkUnit{T}, ::Type{Array}; makepsf::Bool=false) where T
    subgrid = zeros(
        SMatrix{2, 2, Complex{T}, 4}, workunit.subgridspec.Nx, workunit.subgridspec.Ny
    )

    dift!(subgrid, workunit, Val(makepsf))

    subgridflat = reinterpret(reshape, Complex{T}, subgrid)
    fft!(subgridflat, (2, 3))
    subgrid ./= (workunit.subgridspec.Nx * workunit.subgridspec.Ny)
    return fftshift(subgrid)
end

function dift!(subgrid::Matrix{SMatrix{2, 2, Complex{T}, 4}}, workunit::WorkUnit{T}, ::Val{makepsf}) where {T, makepsf}
    lms = fftfreq(workunit.subgridspec.Nx, T(1 / workunit.subgridspec.scaleuv))::Frequencies{T}
    uvdata = workunit.data

    Threads.@threads for idx in CartesianIndices(subgrid)
        lpx, mpx = Tuple(idx)
        l, m = lms[lpx], lms[mpx]

        @simd for i in 1:length(uvdata)
            phase = 2π * 1im * (
                (uvdata.u[i] - workunit.u0) * l +
                (uvdata.v[i] - workunit.v0) * m +
                (uvdata.w[i] - workunit.w0) * ndash(l, m)
            )
            if makepsf
                subgrid[lpx, mpx] += uvdata.weights[i] * exp(phase)
            else
                subgrid[lpx, mpx] += uvdata.weights[i] .* uvdata.data[i] * exp(phase)
            end
        end
        subgrid[lpx, mpx] = (
            workunit.Aleft[lpx, mpx] * subgrid[lpx, mpx] * adjoint(workunit.Aright[lpx, mpx])
        )
    end
end