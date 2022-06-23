function predict!(
    workunits::Vector{WorkUnit{T}},
    img::AbstractMatrix{SMatrix{2, 2, Complex{T}, 4}},
    gridspec::GridSpec,
    taper::Matrix{T},
    subtaper::Matrix{T},
    ::Type{wrapper};
    degridop=degridop_replace
) where {T, wrapper}
    # Inverse tapering
    img = map(img, taper) do val, t
        if iszero(t)
            return zero(val)
        else
            return val / t
        end
    end

    # To avoid unnecessary fftshifts, we now move to 'standard ordering' and
    # do a final fftshift on each wlayer.
    fftshift!(img)
    subtaper = wrapper(ifftshift(subtaper))

    ls = wrapper(fftfreq(gridspec.Nx, 1 / gridspec.scaleuv))
    ms = wrapper(fftfreq(gridspec.Ny, 1 / gridspec.scaleuv))

    wlayerd = wrapper{SMatrix{2, 2, Complex{T}, 4}, 2}(undef, gridspec.Nx, gridspec.Ny)
    wlayerdflat = reinterpret(reshape, Complex{T}, wlayerd)

    t_degrid, t_preprocess = 0., 0.
    for w0 in unique(workunit.w0 for workunit in workunits)
        println("Processing w=$(w0) layer...")

        t_preprocess += @elapsed CUDA.@sync begin
            copy!(wlayerd, img)

            # w-layer decorrection
            map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
                lpx, mpx = Tuple(idx)
                l, m = ls[lpx], ms[mpx]
                return val * exp(-2π * 1im * w0 * ndash(l, m))
            end

            fft!(wlayerdflat, (2, 3))  # inplace

            # Revert back to zero centering the power since extract subgrid
            # and degridder! expect this ordering
            fftshift!(wlayerd)
        end

        t_degrid += @elapsed CUDA.@sync begin
            wworkunits = [wu for wu in workunits if wu.w0 == w0]
            Pigi.degridder!(wworkunits, wlayerd, subtaper, degridop)
        end
    end
    println("Elapsed degridding: $(t_degrid) Elapsed w-layer pre-processing: $(t_preprocess)")
end