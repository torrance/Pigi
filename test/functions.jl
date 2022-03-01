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

function idft!(grid::AbstractMatrix{SMatrix{2, 2, Complex{T}, 4}},  uvdata::StructVector{Pigi.UVDatum{T}}, gridspec::Pigi.GridSpec, normfactor::T; gpu::Bool = true) where T
    if gpu
        gridd = CuArray(grid)
        uvdatad = replace_storage(CuArray, uvdata)

        kernel = @cuda launch=false _idft!(gridd, uvdatad, gridspec, normfactor)
        config = launch_configuration(kernel.fun)
        threads = min(config.threads, length(gridd))
        blocks = cld(length(gridd), threads)
        kernel(gridd, uvdatad, gridspec, normfactor; threads, blocks)

        copyto!(grid, gridd)
    else
        _idft!(grid, uvdata, gridspec, normfactor)
    end
end

function _idft!(dft::Matrix{SMatrix{2, 2, Complex{T}, 4}}, uvdata::StructVector{Pigi.UVDatum{T}}, gridspec::Pigi.GridSpec, normfactor::T) where T
    rowscomplete = 0
    Threads.@threads for lmpx in CartesianIndices(dft)
        lpx, mpx = Tuple(lmpx)
        if lpx == size(dft)[2]
            rowscomplete += 1
            print("\r", rowscomplete / size(dft)[1] * 100)
        end

        l, m = Pigi.px2sky(lpx, mpx, gridspec)
        n = Pigi.ndash(l, m)

        val = zero(SMatrix{2, 2, ComplexF64, 4})
        for uvdatum in uvdata
            val += uvdatum.data * exp(
                2π * 1im * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
            )
        end
        dft[lpx, mpx] = val / normfactor
    end
end

function _idft!(dft::CuDeviceMatrix{SMatrix{2, 2, Complex{T}, 4}}, uvdata::StructVector{Pigi.UVDatum{T}}, gridspec::Pigi.GridSpec, normfactor::T) where T
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(dft)
        lpx, mpx = Tuple(CartesianIndices(dft)[idx])
        l, m = Pigi.px2sky(lpx, mpx, gridspec)
        n = Pigi.ndash(l, m)

        val = zero(SMatrix{2, 2, Complex{T}, 4})
        for uvdatum in uvdata
            val += uvdatum.data * exp(
                2π * 1im * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
            )
        end
        dft[lpx, mpx] = val / normfactor
    end

    return nothing
end

function dft!(uvdata::StructVector{Pigi.UVDatum{T}}, img::Matrix{SMatrix{2, 2, Complex{T}, 4}}, gridspec::Pigi.GridSpec) where T
    idxs = findall(x -> !iszero(x), img)
    for (i, uvdatum) in enumerate(uvdata)
        data = SMatrix{2, 2, Complex{T}, 4}(0, 0, 0, 0)
        for idx in idxs
            lpx, mpx = Tuple(idx)
            l, m = Pigi.px2sky(lpx, mpx, gridspec)
            data += img[lpx, mpx] * exp(
                -2π * 1im * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * Pigi.ndash(l, m))
            )
        end

        uvdata.data[i] = data
    end
end