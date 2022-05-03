function main(;
        imgsize::Int, subgridpadding::Int, mgain::Float64, name::String,
        wrapper::Type{T}, miter::Int, autothreshold::Float64, subgridsize::Int,
        wstep::Int, imgthreshold::Float64, taperthreshold::Float64, msetname::String,
        chanstart::Int, chanstop::Int, scale::Float64, precision::Type{S},
        weight::Symbol, briggsweight::Float64
    ) where {T <: AbstractArray, S <: AbstractFloat}

    # Open measurement set
    print("Reading data into memory... ")
    elapsed = @elapsed begin
        mset = MeasurementSet(
            msetname;
            chanstart=chanstart,
            chanstop=chanstop
        )
        uvdata = StructArray(Pigi.read(mset; precision))
    end
    printfmtln("done. {1:d} uvw samples loaded. Elapsed: {2:.2f} s", length(uvdata), elapsed)


    # Calculate mastergrid size based on taper thresholds
    paddingfactor = taperpadding(imgthreshold, taperthreshold)
    printfmtln("Using padding factor: {:.2f}", paddingfactor)

    masterpadding = (nextfastfft(round(Int, paddingfactor * imgsize)) - imgsize) ÷ 2
    gridspec = GridSpec(imgsize + 2 * masterpadding, imgsize + 2 * masterpadding, scalelm=deg2rad(scale / 3600))
    println("Using mastergrid size: $(gridspec.Nx) x $(gridspec.Ny)")

    taper = mkkbtaper(gridspec; threshold=taperthreshold)


    # Weight data
    print("Creating imageweighter... ")
    elapsed = @elapsed if weight == :uniform
        imageweighter = Pigi.Uniform(uvdata, gridspec)
    elseif weight == :natural
        imageweighter = Pigi.Natural(uvdata, gridspec)
    elseif weight == :briggs
        imageweighter = Pigi.Briggs(uvdata, gridspec, briggsweight)
    else
        throw(ArgumentError("weight must be: [:uniform | :natural | :briggs]"))
    end
    printfmtln("done. Elapsed {:.2f} s", elapsed)

    print("Weighting data... ")
    elapsed = @elapsed Pigi.applyweights!(uvdata, imageweighter)
    printfmtln("done. Elapsed: {:.2f} s", elapsed)


    # Partition data
    print("Partitioning data... ")
    subgridspec = Pigi.GridSpec(subgridsize, subgridsize, scaleuv=gridspec.scaleuv)
    elapsed = @elapsed workunits = Pigi.partition(uvdata, gridspec, subgridspec, subgridpadding, wstep, taper)
    printfmtln("done. Elapsed {:.2f} s", elapsed)

    occupancy = [length(workunit.data) for workunit in workunits]
    printfmtln(
        "WorkUnits: {:d} Occupancy (min/mean/median/max): {:d}/{:.1f}/{:.1f}/{:d}",
        length(workunits), minimum(occupancy), mean(occupancy), median(occupancy), maximum(occupancy)
    )


    # Make PSF if we are cleaning
    if miter > 0
        println("Making psf... ")
        elapsed = @elapsed begin
            psf = Pigi.invert(workunits, gridspec, taper, wrapper, makepsf=true)
            psf = @view psf[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
            psf_stokesI = map(psf) do x
                (real(x[1, 1]) + real(x[2, 2])) / 2
            end

            FITS("$(name)-psf.fits", "w") do f
                write(f, psf_stokesI)
            end
        end
        printfmtln("PSF done. Elapsed {:.2f} s", elapsed)

        # Clip PSF size for cleaning and fitting
        elapsed = @elapsed psf_stokesI_clipped = Pigi.psfclip(psf_stokesI, 0.2)
        printfmtln("PSF clipped to $(size(psf_stokesI_clipped)) (original $(size(psf_stokesI)). Elapsed {:.2f} s", elapsed)

        elapsed = @elapsed (xsigma, ysigma, pa), fitted = Pigi.psffit(psf_stokesI_clipped)
        printfmtln("Fitted PSF: $((xsigma, ysigma, pa)). Elapsed {:.2f} s", elapsed)

        # Make fitted PSF for later use in convolution
        print("Making fitted psf image... ")
        psf_fitted = zeros(imgsize, imgsize)
        elapsed = @elapsed mkpsffitted!(psf_fitted, xsigma, ysigma, pa)
        printfmtln("done. Elapsed {:.2f} s", elapsed)
    end


    # Perform major clean iterations
    components = zeros(imgsize, imgsize)
    mcomponentspadded = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    mcomponentspadded_flat = reinterpret(reshape, Complex{precision}, mcomponentspadded)

    for imajor in 1:miter
        println("\nStarting major iteration $(imajor)")
        println("Making dirty image...")
        elapsed = @elapsed dirty = Pigi.invert(workunits, gridspec, taper, wrapper)
        printfmtln("Dirty image done. Elapsed {:.2f} s", elapsed)
        GC.gc(true)

        # Clean only over smaller map
        dirty = @view dirty[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
        residual_stokesI = map(dirty) do x
            (real(x[1, 1]) + real(x[2, 2])) / 2
        end

        if imajor == 1
            FITS("$(name)-dirty.fits", "w") do f
                write(f, residual_stokesI)
            end
        end

        # Estimate residual noise using median absolute deviation from median
        # which we use to set cleaning thresholds
        med = median(residual_stokesI)
        noise = 1.4826 * median(abs(x) - med for x in residual_stokesI)
        println("Peak map value: $(maximum(abs, residual_stokesI)) Residual map noise: $(noise)")

        # Do the actual cleaning
        println("Cleaning...")
        elapsed = @elapsed mcomponents, niters = Pigi.clean!(
            residual_stokesI, psf_stokesI_clipped;
            mgain, threshold=autothreshold * noise
        )
        components += mcomponents
        GC.gc(true)
        printfmtln("Cleaning done. Elapsed {:.2f} s", elapsed)

        peakresidual = maximum(abs, residual_stokesI)
        println("Residual peak: $(peakresidual)")

        # We cleaned on a smaller stokes I map with no padding. Copy the stokes I to a full padded instrumental array.
        print("Building padded component image... ")
        elapsed = @elapsed begin
            fill!(mcomponentspadded, zero(eltype(mcomponentspadded)))
            mcomponentspadded_flat[1, 1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding] .= mcomponents
            mcomponentspadded_flat[4, 1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding] .= mcomponents
        end
        printfmtln("done. Elapsed {:.2f}", elapsed)

        # Predict clean components into visibilities and subtract
        @time Pigi.predict!(workunits, mcomponentspadded, gridspec, taper, wrapper, degridop=Pigi.degridop_subtract)
        GC.gc(true);

        if peakresidual <= autothreshold * noise
            println("Clean threshold reached: finishing cleaning")
            @goto cleaningfinished
        end
    end
    println("Cleaning terminated due to maximum major iteration limit")
    @label cleaningfinished


    # Perform final inversion
    println("\nFinal inversion...")
    elapsed = @elapsed dirty = Pigi.invert(workunits, gridspec, taper, wrapper)
    println("Final inversion done. Elapsed {:.2f} s", elapsed)


    dirty = @view dirty[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
    residual_stokesI = map(dirty) do x
        (real(x[1, 1]) + real(x[2, 2])) / 2
    end


    # Write out FITS images for model, residual and (if applicable) restored images
    FITS("$(name)-model.fits", "w") do f
        write(f, real.(components))
    end

    FITS("$(name)-residual.fits", "w") do f
        write(f, real.(residual_stokesI))
    end

    if miter > 0
        # Restore clean components
        print("Creating restored image...")
        x0, y0 = size(psf_fitted) .÷ 2
        elapsed = @elapsed restored_stokesI = conv(components, psf_fitted)[1 + x0:end - x0 + 1, 1 + y0:end - y0 + 1] .+ residual_stokesI
        printfmtln("done. Elapsed {:.2f} s", elapsed)

        FITS("$(name)-restored.fits", "w") do f
            write(f, real.(restored_stokesI))
        end
    end
end