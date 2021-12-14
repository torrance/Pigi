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

    wlayers = Tuple{Int, Matrix{SMatrix{2, 2, Complex{T}, 4}}}[]
    mastergrid = Array{SMatrix{2, 2, Complex{T}, 4}, 2}(undef, gridspec.Nx, gridspec.Ny)

    gridded = Threads.Atomic{Int}(0)
    for w0 in unique(subgrid.w0 for subgrid in subgrids)
        fill!(mastergrid, zero(SMatrix{2, 2, Complex{T}, 4}))

        for subgrid in subgrids
            if subgrid.w0 == w0
                grid = gridder(subgrid, makepsf=makepsf)
                addsubgrid!(mastergrid, grid, subgrid)

                Threads.atomic_add!(gridded, 1)
                print("\rGridded $(gridded[])/$(length(subgrids)) subgrids...")
            end
        end

        # Switch to standard ordering do that we can do the fft
        # We do this now rather than in parallel, since this results in a copy of memory
        mastergridshifted = ifftshift(mastergrid)
        push!(wlayers, (w0, mastergridshifted))

        if length(wlayers) == Threads.nthreads()
            print("\nFFT'ing a batch of w-layers...")
            Threads.@threads for (w0, wlayer) in wlayers
                addcorrectedwlayer!(img, imglock, w0, wlayer, plan, ls, ms)
            end
            empty!(wlayers)
            println(" Done.")
        end
    end

    print("\nFFT'ing remaining w-layers...")
    Threads.@threads for (w0, wlayer) in wlayers
        addcorrectedwlayer!(img, imglock, w0, wlayer, plan, ls, ms)
    end
    empty!(wlayers)
    println("Done")

    # Shift back to zero-centering order.
    img = fftshift(img)

    # Our final image still has a taper applied, time to remove it.
    removetaper!(img, gridspec, taper)

    println("Inversion complete")
    return img
end

function addcorrectedwlayer!(
    img::Matrix{SMatrix{2, 2, Complex{T}, 4}},
    imglock,
    w0,
    mastergrid::Matrix{SMatrix{2, 2, Complex{T}, 4}},
    plan,
    ls,
    ms
) where {T}
    # Create a flattened view into the data so that we can fft it.
    mastergridflat = reinterpret(reshape, Complex{T}, mastergrid)
    plan * mastergridflat  # inplace fft

    # Apply w-correction
    Threads.@threads for idx in CartesianIndices(mastergrid)
        lpx, mpx = Tuple(idx)
        l, m = ls[lpx], ms[mpx]
        mastergrid[lpx, mpx] *= exp(2π * 1im * w0 * ndash(l, m)) * length(mastergrid)
    end

    # Combine with w-layers
    lock(imglock)
        img .+= mastergrid
    unlock(imglock)
end