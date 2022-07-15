function invert(
    ::Type{S},
    workunits::Vector{WorkUnit{T}},
    gridspec::GridSpec,
    taper::Matrix{T},
    subtaper::Matrix{T},
    ::Type{wrapper};
    makepsf=false
) where {T, S <: OutputType{T}, wrapper}

    t_grid, t_postprocess = 0., 0.

    img = zeros(S, gridspec.Nx, gridspec.Ny)

    # Create the iterators for standard ordering (used to apply w-correction)
    ls = fftfreq(gridspec.Nx, T(1 / gridspec.scaleuv))
    ms = fftfreq(gridspec.Ny, T(1 / gridspec.scaleuv))

    wlayer = CUDA.Mem.pin(Array{S}(undef, gridspec.Nx, gridspec.Ny))
    wlayerd = wrapper{S}(undef, gridspec.Nx, gridspec.Ny)

    # Use standard ordering for taper
    subtaper = wrapper(ifftshift(subtaper))

    for w0 in unique(wu.w0 for wu in workunits)
        println("Processing w=$(w0) layer...")

        t_grid += @elapsed CUDA.@sync begin
            fill!(wlayerd, zero(S))
            wworkunits = [wu for wu in workunits if wu.w0 == w0]
            gridder!(wlayerd, wworkunits, subtaper; makepsf)
        end

        t_postprocess += @elapsed begin
            fftshift!(wlayerd)
            ifft!(wlayerd)

            # w-layer correction
            map!(wlayerd, wlayerd, CartesianIndices(wlayerd)) do val, idx
                lpx, mpx = Tuple(idx)
                l, m = ls[lpx], ms[mpx]
                return val * exp(2im * T(π) * w0 * ndash(l, m)) * length(wlayerd)
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
    @time img ./= taper

    println("Inversion complete")
    return img
end