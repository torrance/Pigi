function invert(subgrids::Vector{Subgrid{T}}, gridspec::GridSpec, taper; makepsf=false) where T
    # Create the fftplan just once, for a slight performance win.
    # We use img as a proxy for the planning, since it has the same shape and type
    # as the mastergrids.
    img = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)
    imgflat = reinterpret(reshape, Complex{T}, img)
    plan = plan_ifft!(imgflat, (2, 3))
    imglock = ReentrantLock()

    # Create the iterators for standard ordering (used to apply w-correction)
    ls = fftfreq(gridspec.Nx, 1 / gridspec.scaleuv)
    ms = fftfreq(gridspec.Ny, 1 / gridspec.scaleuv)

    # Multithreading notes
    # We want to evenly distribute the subgrids across all cores, however, we cannot
    # process all w-layers at once, as each w-layer requires a (potentially) large allocation
    # to store its results. We thus batch the w layers. However, simply batching w-layers
    # might not be optimal, since we either a) might be approximately planar or b) the
    # subgrids may be poorly distributed amongst w layers.

    gridded = Threads.Atomic{Int}(0)
    Threads.@threads for w0 in unique(subgrid.w0 for subgrid in subgrids)
        wsubgrids = filter(sg -> sg.w0 == w0, subgrids)
        mastergrid = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)

        for subgrid in wsubgrids
            grid = gridder(subgrid, makepsf=makepsf)
            addsubgrid!(mastergrid, grid, subgrid)

            Threads.atomic_add!(gridded, 1)
            print("\rGridded $(gridded[])/$(length(subgrids)) subgrids...")
        end

        # Switch to standard ordering do that we can do the fft
        mastergrid = ifftshift(mastergrid)

        # Create a flattened view into the data so that we can fft it.
        mastergridflat = reinterpret(reshape, Complex{T}, mastergrid)
        plan * mastergridflat  # inplace fft

        # Apply w-correction
        for (mpx, m) in enumerate(ms), (lpx, l) in enumerate(ls)
            mastergrid[lpx, mpx] *= exp(2π * 1im * w0 * ndash(l, m)) * length(mastergrid)
        end

        # Combine with w-layers
        lock(imglock)
            img .+= mastergrid
        unlock(imglock)
    end
    println("\nDone.")

    # Shift back to zero-centering order.
    img = fftshift(img)

    # Our final image still has a taper applied, time to remove it.
    removetaper!(img, gridspec, taper)

    return img
end