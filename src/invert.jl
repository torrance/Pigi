function invert(workunits::Vector{WorkUnit{T}}, gridspec::GridSpec, taper, ::Type{wrapper}; makepsf=false) where {T, wrapper}
    # Create the fftplan just once, for a slight performance win.
    # We use img as a proxy for the planning, since it has the same shape and type
    # as the mastergrids.
    img = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)

    # Create the iterators for standard ordering (used to apply w-correction)
    ls = wrapper(fftfreq(gridspec.Nx, 1 / gridspec.scaleuv))
    ms = wrapper(fftfreq(gridspec.Ny, 1 / gridspec.scaleuv))

    sort!(workunits, rev=true, by=x -> length(x.data))

    subgrids = Vector{Matrix{SMatrix{2, 2, Complex{T}, 4}}}(undef, length(workunits))

    gridded = Threads.Atomic{Int}(0)
    @time Base.@sync for (i, workunit) in enumerate(workunits)
        Base.@async begin
            subgrids[i] = gridder(workunit, wrapper, makepsf=makepsf)
            Threads.atomic_add!(gridded, 1)
            print("\rGridded $(gridded[])/$(length(workunits)) workunits...")
        end
    end

    wlayer = Array{SMatrix{2, 2, Complex{T}, 4}}(undef, gridspec.Nx, gridspec.Ny)
    wlayerd = wrapper{SMatrix{2, 2, Complex{T}, 4}}(undef, gridspec.Nx, gridspec.Ny)

    wlayerdflat = reinterpret(reshape, Complex{T}, wlayerd)
    plan = plan_ifft!(wlayerdflat, (2, 3))

    @time for w0 in unique(wu.w0 for wu in workunits)
        println("Processing w=$(w0) layer...")

        fill!(wlayer, zero(SMatrix{2, 2, Complex{T}, 4}))
        for (subgrid, workunit) in zip(subgrids, workunits)
            if workunit.w0 == w0
                addsubgrid!(wlayer, subgrid, workunit)
            end
        end

        fftshift!(wlayer)
        copy!(wlayerd, wlayer)

        plan * wlayerdflat

        # w-layer correction
        map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
            lpx, mpx = Tuple(idx)
            l, m = ls[lpx], ms[mpx]
            return val * exp(2π * 1im * w0 * ndash(l, m)) * length(wlayerd)
        end

        copy!(wlayer, wlayerd)
        img .+= wlayer
    end

    # Shift back to zero-centering order.
    fftshift!(img)

    # Our final image still has a taper applied, time to remove it.
    println("Removing taper...")
    @time removetaper!(img, gridspec, taper)

    println("Inversion complete")
    return img
end