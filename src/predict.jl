function predict!(
    subgrids::Vector{Subgrid{T}},
    img::Matrix{SMatrix{2, 2, Complex{T}, 4}},
    gridspec::GridSpec,
    taper
) where {T}
    # To avoid unnecessary fftshifts, we now move to 'standard ordering' and
    # do a final fftshift on each wlayer.
    img = ifftshift(img)
    ls = fftfreq(gridspec.Nx, 1 / gridspec.scaleuv)
    ms = fftfreq(gridspec.Ny, 1 / gridspec.scaleuv)

    # Inverse tapering
    Threads.@threads for (mpx, m) in collect(enumerate(ms))
        for (lpx, l) in enumerate(ls)
            # The prediction map is often sparse, so skip exhaustive application
            # of taper.
            if iszero(img[lpx, mpx])
                continue
            end

            tlm = taper(l, m)
            if iszero(tlm)
                img[lpx, mpx] = zero(SMatrix{2, 2, Complex{T}, 4})
            else
                img[lpx, mpx] /= tlm
            end
        end
    end

    # Create the fftplan just once, for a slight performance win.
    # We use img as a proxy for the planning, since it has the same shape and type
    # as the mastergrids.
    imgflat = reinterpret(reshape, Complex{T}, img)
    plan = plan_fft!(imgflat, (2, 3))

    gridded = Threads.Atomic{Int}(0)
    Threads.@threads for w0 in unique(subgrid.w0 for subgrid in subgrids)
        mastergrid = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)

        # Apply w-layer (de)correction
        for (mpx, m) in enumerate(ms), (lpx, l) in enumerate(ls)
            mastergrid[lpx, mpx] = img[lpx, mpx] * exp(-2π * 1im * w0 * ndash(l, m))
        end

        # Flatten the data structure so that we can FFT
        mastergridflat = reinterpret(reshape, Complex{T}, mastergrid)
        plan * mastergridflat  # inplace

        # Revert back to zero centering the power
        mastergrid = fftshift(mastergrid)

        for subgrid in subgrids
            if subgrid.w0 == w0
                visgrid = Pigi.extractsubgrid(mastergrid, subgrid)
                Pigi.degridder!(subgrid, visgrid)
                Threads.atomic_add!(gridded, 1)
                print("\rDegridded $(gridded[])/$(length(subgrids)) subgrids...")
            end
        end
    end
    println("\nDone.")
end