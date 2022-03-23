function invert(workunits::Vector{WorkUnit{T}}, gridspec::GridSpec, taper, ::Type{wrapper}; makepsf=false) where {T, wrapper}
    t_grid, t_postprocess = 0., 0.

    # Create the fftplan just once, for a slight performance win.
    # We use img as a proxy for the planning, since it has the same shape and type
    # as the mastergrids.
    img = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)

    # Create the iterators for standard ordering (used to apply w-correction)
    ls = wrapper(fftfreq(gridspec.Nx, 1 / gridspec.scaleuv))
    ms = wrapper(fftfreq(gridspec.Ny, 1 / gridspec.scaleuv))

    wlayer = CUDA.Mem.pin(Array{SMatrix{2, 2, Complex{T}, 4}}(undef, gridspec.Nx, gridspec.Ny))
    wlayerd = wrapper{SMatrix{2, 2, Complex{T}, 4}}(undef, gridspec.Nx, gridspec.Ny)
    wlayerdflat = reinterpret(reshape, Complex{T}, wlayerd)

    for w0 in unique(wu.w0 for wu in workunits)
        println("Processing w=$(w0) layer...")

        t_grid += @elapsed CUDA.@sync begin
            fill!(wlayerd, zero(SMatrix{2, 2, Complex{T}, 4}))
            wworkunits = [wu for wu in workunits if wu.w0 == w0]
            gridder!(wlayerd, wworkunits; makepsf)
        end

        t_postprocess += @elapsed begin
            fftshift!(wlayerd)
            ifft!(wlayerdflat, (2, 3))

            # w-layer correction
            map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
                lpx, mpx = Tuple(idx)
                l, m = ls[lpx], ms[mpx]
                return val * exp(2π * 1im * w0 * ndash(l, m)) * length(wlayerd)
            end

            copy!(wlayer, wlayerd)
            img .+= wlayer
        end
    end
    println("Elapsed gridding: $(t_grid) Elapsed w-layer post-processing: $(t_postprocess)")

    # Shift back to zero-centering order.
    fftshift!(img)

    # Our final image still has a taper applied, time to remove it.
    println("Removing taper...")
    @time removetaper!(img, gridspec, taper)

    println("Inversion complete")
    return img
end