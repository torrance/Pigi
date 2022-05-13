function main(;
        imgsize::Int, subgridpadding::Int, mgain::Float64, name::String,
        wrapper::Type{T}, miter::Int, autothreshold::Float64, subgridsize::Int,
        wstep::Int, imgthreshold::Float64, taperthreshold::Float64, msetname::String,
        chanstart::Int, chanstop::Int, scale::Float64, precision::Type{S},
        weight::Symbol, briggsweight::Float64, channelsout::Int
    ) where {T <: AbstractArray, S <: AbstractFloat}

    # Open full measurement to get metadata and calculate weights
    mset = MeasurementSet(
        msetname;
        chanstart=chanstart,
        chanstop=chanstop
    )
    printfmtln(
        "Measurement set(s) opened with {:d} rows, {:d} channels ({:.2f} - {:.2f} MHz)",
        mset.nrows, mset.nchans, mset.freqs[1] / 1e6, mset.freqs[end] / 1e6
    )


    # Calculate mastergrid size based on taper thresholds
    paddingfactor = taperpadding(imgthreshold, taperthreshold)
    printfmtln("Using padding factor: {:.2f}", paddingfactor)

    masterpadding = (nextfastfft(round(Int, paddingfactor * imgsize)) - imgsize) ÷ 2
    gridspec = GridSpec(imgsize + 2 * masterpadding, imgsize + 2 * masterpadding, scalelm=deg2rad(scale / 3600))
    subgridspec = Pigi.GridSpec(subgridsize, subgridsize, scaleuv=gridspec.scaleuv)
    println("Using mastergrid size: $(gridspec.Nx) x $(gridspec.Ny)")

    taper = mkkbtaper(gridspec; threshold=taperthreshold)


    # Weight data
    print("Creating imageweighter... ")
    uvdata = read(mset; precision)
    elapsed = @elapsed if weight == :uniform
        imageweighter = Uniform(precision, uvdata, gridspec)
    elseif weight == :natural
        imageweighter = Natural(precision, uvdata, gridspec)
    elseif weight == :briggs
        imageweighter = Briggs(precision, uvdata, gridspec, briggsweight)
    else
        throw(ArgumentError("weight must be: [:uniform | :natural | :briggs]"))
    end
    printfmtln("done. Elapsed {:.2f} s", elapsed)


    # Calculate distribution of work and initialize workers
    channelwidth = cld(mset.nchans, channelsout)
    channeledges = [
        ((i - 1) * channelwidth + 1, min(i * channelwidth, mset.nchans))
        for i in 1:channelsout
    ]
    midfreqs = [mean(mset.freqs[chstart:chstop]) for (chstart, chstop) in channeledges]
    printfmtln("Distributing work across {:d} workers:", channelsout)
    for (chstart, chstop) in channeledges
        printfmtln(
            "  {:.2f} - {:.2f} MHz ({:d} channels)",
            mset.freqs[chstart] / 1e6, mset.freqs[chstop] / 1e6, chstop - chstart + 1
        )
    end

    # Assign work to fixed processes, keeping data locallly for the lifetime of the program
    assignedworkers = AssignedWorkers(
        first(Iterators.cycle(workers()), channelsout)  # workid => pid
    )


    submsets, workunitss, psf_stokesIs = zip(
        pmap(assignedworkers, channeledges) do (chanstart, chanend)
            println("Running workid: $((chanstart, chanend)) on pid: $(myid())")

            GC.gc(true); CUDA.reclaim(); CUDA.memory_status(); println()

            # Open measurement set
            print("Reading data into memory... ")
            elapsed = @elapsed begin
                submset = MeasurementSet(msetname; chanstart, chanstop)
                uvdata = StructArray(Pigi.read(submset; precision))
            end
            printfmtln("done. {1:d} uvw samples loaded. Elapsed: {2:.2f} s", length(uvdata), elapsed)


            print("Weighting data... ")
            elapsed = @elapsed Pigi.applyweights!(uvdata, imageweighter)
            printfmtln("done. Elapsed: {:.2f} s", elapsed)


            # Partition data
            print("Partitioning data... ")
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

                    # FITS(format("{:s}-psf-{:04d}.fits", name, workid), "w") do f
                    #     write(f, psf_stokesI)
                    # end
                    # TODO: Should this be grouped on master to coadd timesteps?
                end
                printfmtln("PSF done. Elapsed {:.2f} s", elapsed)
            end

            return RemoteRef(submset), RemoteRef(workunits), psf_stokesI
        end
    ...)

    psf_stokesI_MFS = mean(psf_stokesIs)

    # Rearrange memory layout of psf_stokesIs to SVector{N} for each channelout,
    # in preparation for cleaning.
    elapsed = @elapsed print("Rejigging psf_stokesIs memory layout... ")
    psf_stokesIs = permute2vector(psf_stokesIs)
    printfmtln("done. Elapsed {:.2f} s", elapsed)


    # Perform major clean iterations
    components = zeros(SVector{channelsout, precision}, imgsize, imgsize)
    mcomponentspadded = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny, channelsout)
    mcomponentspadded_flat = reinterpret(reshape, Complex{precision}, mcomponentspadded)

    for imajor in 1:miter
        println("\nStarting major iteration $(imajor)")

        residual_stokesIs = pmap(assignedworkers, workunitss) do workunits
            GC.gc(true); CUDA.reclaim(); CUDA.memory_status(); println()

            println("Making dirty image...")
            elapsed = @elapsed dirty = Pigi.invert(workunits[], gridspec, taper, wrapper)
            printfmtln("Dirty image done. Elapsed {:.2f} s", elapsed)
            GC.gc(true)

            # Clean only over smaller map
            dirty = @view dirty[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
            residual_stokesI = map(dirty) do x
                (real(x[1, 1]) + real(x[2, 2])) / 2
            end

            return residual_stokesI
        end

        elapsed = @elapsed print("Rejigging residual_stokesIs memory layout... ")
        residual_stokesI_MFS = mean(residual_stokesIs)
        residual_stokesIs = permute2vector(residual_stokesIs)
        printfmtln("done. Elapsed {:.2f} s", elapsed)

        if imajor == 1
            FITS("$(name)-MFS-dirty.fits", "w") do f
                write(f, residual_stokesI_MFS)
            end
        end

        # Estimate residual noise using median absolute deviation from median
        # which we use to set cleaning thresholds
        med = median(residual_stokesI_MFS)
        noise = 1.4826 * median(abs(x) - med for x in residual_stokesI_MFS)
        println("Peak map value: $(maximum(abs, residual_stokesI_MFS)) Residual map noise: $(noise)")

        # Do the actual cleaning
        println("Cleaning...")
        elapsed = @elapsed mcomponents, niters = Pigi.clean!(
            residual_stokesIs, psf_stokesIs, midfreqs;
            mgain, threshold=autothreshold * noise
        )
        components += mcomponents
        GC.gc(true)
        printfmtln("Cleaning done. {:.2f} Jy removed this cycle. Elapsed {:.2f} s", sum(mean, mcomponents), elapsed)

        # We cleaned on a smaller stokes I map with no padding. Copy the stokes I to a full padded instrumental array.
        print("Building padded component image... ")
        fill!(mcomponentspadded, zero(eltype(mcomponentspadded)))
        mcomponents_flat = reshape(reinterpret(eltype(eltype(mcomponents)), mcomponents), channelsout, imgsize, imgsize)
        elapsed = @elapsed for i in 1:channelsout
            mcomponentspadded_flat[1, 1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding, i] .= mcomponents_flat[i, :, :]
            mcomponentspadded_flat[4, 1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding, i] .= mcomponents_flat[i, :, :]
        end
        printfmtln("done. Elapsed {:.2f}", elapsed)

        pmap(assignedworkers, workunitss, eachslice(mcomponentspadded, dims=3)) do workunits, mcomponentspadded
            GC.gc(true); CUDA.reclaim(); CUDA.memory_status(); println()

            # Predict clean components into visibilities and subtract
            Pigi.predict!(workunits[], mcomponentspadded, gridspec, taper, wrapper, degridop=Pigi.degridop_subtract)
            return nothing
        end

        if maximum(abs ∘ mean, residual_stokesIs) <= autothreshold * noise
            println("Clean threshold reached: finishing cleaning")
            @goto cleaningfinished
        end
    end
    println("Cleaning terminated due to maximum major iteration limit")
    @label cleaningfinished


    # Perform final inversion
    println("\nFinal inversion...")
    elapsed = @elapsed residual = pmap(assignedworkers, workunitss) do workunits
        GC.gc(true); CUDA.reclaim(); CUDA.memory_status(); println()

        return Pigi.invert(workunits[], gridspec, taper, wrapper)
    end
    printfmtln("Final inversion done. Elapsed {:.2f} s", elapsed)

    residual_MFS = sum(residual)
    residual_MFS = @view residual_MFS[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
    residual_stokesI = map(residual_MFS) do x
        (real(x[1, 1]) + real(x[2, 2])) / 2
    end

    # Write out FITS images for model, residual and (if applicable) restored images
    components_MFS = map(mean, components)
    FITS("$(name)-MFS-model.fits", "w") do f
        write(f, components_MFS)
    end

    FITS("$(name)-MFS-residual.fits", "w") do f
        write(f, real.(residual_stokesI))
    end

    if miter > 0
        psf_MFS = map(mean, psf_stokesIs)

        # Clip PSF size for cleaning and fitting
        elapsed = @elapsed psf_MFS_clipped = Pigi.psfclip(psf_MFS, 0.2)
        printfmtln("PSF clipped to $(size(psf_MFS_clipped)) (original $(size(psf_MFS)). Elapsed {:.2f} s", elapsed)

        elapsed = @elapsed (xsigma, ysigma, pa), fitted = Pigi.psffit(psf_MFS_clipped)
        printfmtln("Fitted PSF: $((xsigma, ysigma, pa)). Elapsed {:.2f} s", elapsed)

        # # Make fitted PSF for use in convolution
        print("Making fitted psf image... ")
        psf_fitted = zeros(imgsize, imgsize)
        elapsed = @elapsed mkpsffitted!(psf_fitted, xsigma, ysigma, pa)
        printfmtln("done. Elapsed {:.2f} s", elapsed)

        # Restore clean components
        print("Creating restored image...")
        x0, y0 = size(psf_fitted) .÷ 2
        elapsed = @elapsed restored_stokesI = conv(components_MFS, psf_fitted)[1 + x0:end - x0 + 1, 1 + y0:end - y0 + 1] .+ residual_stokesI
        printfmtln("done. Elapsed {:.2f} s", elapsed)

        FITS("$(name)-MFS-restored.fits", "w") do f
            write(f, real.(restored_stokesI))
        end
    end
end