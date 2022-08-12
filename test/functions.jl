function convolutionalsample!(grid, gridspec, uvdata, kernel, width; uoffset=0, voffset=0)
    for uvdatum in uvdata
        upx, vpx = Pigi.lambda2px(uvdatum.u - uoffset, uvdatum.v - voffset, gridspec)

        for ypx in axes(grid, 2)
            dypx = abs(ypx - vpx)
            if 0 <= dypx <= width
                for xpx in axes(grid, 1)
                    dxpx = abs(xpx - upx)
                    if 0 <= dxpx <= width
                        grid[xpx, ypx] += kernel(sqrt(dxpx^2 + dypx^2)) * uvdatum.weights .* uvdatum.data
                    end
                end
            end
        end
    end
end

function idft!(grid::AbstractMatrix{SMatrix{2, 2, Complex{T}, 4}},  uvdata::StructVector{Pigi.UVDatum{T}}, gridspec::Pigi.GridSpec, normfactor::T, ::Type{wrapper}) where {T, wrapper}
    @kernel function _idft!(dft::AbstractMatrix{SMatrix{2, 2, Complex{T}, 4}}, uvdata::StructVector{Pigi.UVDatum{T}}, gridspec::Pigi.GridSpec, normfactor::T) where T
        idx = @index(Global)
        lpx, mpx = Tuple(CartesianIndices(dft)[idx])

        l, m = Pigi.px2sky(lpx, mpx, gridspec)
        n = Pigi.ndash(l, m)

        val = zero(SMatrix{2, 2, Complex{T}, 4})
        for uvdatum in uvdata
            val += uvdatum.data * exp(
                2im * T(π) * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
            )
        end
        dft[lpx, mpx] = val / normfactor
    end

    # Copy data to wrapper device. If this is Array, this is a noop.
    gridd = wrapper(grid)
    uvdatad = replace_storage(wrapper, uvdata)

    kernel = _idft!(Pigi.kernelconf(gridd)...)
    wait(
        kernel(gridd, uvdatad, gridspec, normfactor; ndrange=length(grid))
    )

    # Copy result back from device.
    copyto!(grid, gridd)
end

function dft!(uvdata::StructVector{Pigi.UVDatum{T}}, img::AbstractMatrix{S}, gridspec::Pigi.GridSpec) where {T, S}
    idxs = findall(x -> !iszero(x), img)
    for (i, uvdatum) in enumerate(uvdata)
        data = zero(S)
        for idx in idxs
            lpx, mpx = Tuple(idx)
            l, m = Pigi.px2sky(lpx, mpx, gridspec)
            l, m = T(l), T(m)

            data += img[lpx, mpx] * exp(
                -2im * T(π) * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * Pigi.ndash(l, m))
            )
        end

        uvdata.data[i] = data
    end
end

function fakebeam(gridspec; l0=0., m0=0.)
    delta1l = (gridspec.Nx ÷ 2) * gridspec.scalelm * 0.25
    delta1m = (gridspec.Ny ÷ 2) * gridspec.scalelm * 0.3

    delta2l = (gridspec.Nx ÷ 2) * gridspec.scalelm * 0.35
    delta2m = (gridspec.Ny ÷ 2) * gridspec.scalelm * 0.4


    return map(CartesianIndices((1:gridspec.Nx, 1:gridspec.Ny))) do lmpx
        lpx, mpx = Tuple(lmpx)
        l, m = Pigi.px2sky(lpx, mpx, gridspec)

        J = Pigi.Comp2x2{Float64}(
            sqrt(exp(-(l - l0)^2 / delta1l^2) * exp(-(m - m0)^2 / delta1m^2)), 0,
            0, sqrt(exp(-(l - l0)^2 / delta2l^2) * exp(-(m - m0)^2 / delta2m^2))
        )

        J *= Pigi.Real2x2{Float64}(cos(l * m), -sin(l * m), sin(l * m), cos(l * m))

        return J
    end
end
