function predict!(
    workunits::Vector{WorkUnit{T}},
    img::AbstractMatrix{S},
    gridspec::GridSpec,
    taper::Matrix{T},
    subtaper::Matrix{T},
    ::Type{wrapper};
    degridop=degridop_replace
) where {T, S <: OutputType{T}, wrapper}
    # Apply inverse taper (with implicit copy so we don't modify img)
    img = map(img, taper) do val, t
        if iszero(t)
            return zero(val)
        else
            return val / t
        end
    end
    img::AbstractMatrix{S}

    # To avoid unnecessary fftshifts, we now move to 'standard ordering' and
    # do a final fftshift on each wlayer.
    fftshift!(img)
    subtaper = wrapper(ifftshift(subtaper))  # Implicit copy: subtaper is not modified

    ls = fftfreq(gridspec.Nx, T(1 / gridspec.scaleuv))
    ms = fftfreq(gridspec.Ny, T(1 / gridspec.scaleuv))

    wlayerd = wrapper{S}(undef, gridspec.Nx, gridspec.Ny)

    t_degrid, t_preprocess = 0., 0.
    for w0 in unique(workunit.w0 for workunit in workunits)
        println("Processing w=$(w0) layer...")

        t_preprocess += @elapsed CUDA.@sync begin
            copy!(wlayerd, img)

            # w-layer decorrection
            map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
                lpx, mpx = Tuple(idx)
                l, m = ls[lpx], ms[mpx]
                return val * exp(-2im * T(π) * w0 * ndash(l, m))
            end

            fft!(wlayerd)

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