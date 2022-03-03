function predict!(
    workunits::Vector{WorkUnit{T}},
    img::Matrix{SMatrix{2, 2, Complex{T}, 4}},
    gridspec::GridSpec,
    taper,
    ::Type{wrapper};
    degridop=degridop_replace
) where {T, wrapper}
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

    lsd = wrapper(ls)
    msd = wrapper(ms)

    wlayerd = wrapper{SMatrix{2, 2, Complex{T}, 4}, 2}(undef, gridspec.Nx, gridspec.Ny)
    wlayer = Array{SMatrix{2, 2, Complex{T}, 4}, 2}(undef, gridspec.Nx, gridspec.Ny)

    wlayerdflat = reinterpret(reshape, Complex{T}, wlayerd)
    plan = plan_fft!(wlayerdflat, (2, 3))

    gridded = Threads.Atomic{Int}(0)
    Base.@sync for w0 in unique(workunit.w0 for workunit in workunits)
        copy!(wlayerd, img)

        # w-layer decorrection
        map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
            lpx, mpx = Tuple(idx)
            l, m = lsd[lpx], msd[mpx]
            return val * exp(-2π * 1im * w0 * ndash(l, m))
        end

        plan * wlayerdflat  # inplace

        copy!(wlayer, wlayerd)

        # Revert back to zero centering the power since extract subgrid
        # and degridder! expect this ordering
        fftshift!(wlayer)

        for workunit in workunits
            if workunit.w0 == w0
                visgrid = Pigi.extractsubgrid(wlayer, workunit)
                Base.@async begin
                    Pigi.degridder!(workunit, visgrid, degridop, wrapper)
                    Threads.atomic_add!(gridded, 1)
                    print("\rDegridded $(gridded[])/$(length(workunits)) workunits...")
                end
            end
        end
    end
    println("\nDone.")
end