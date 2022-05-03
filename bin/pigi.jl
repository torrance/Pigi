#! /usr/bin/env -S julia --threads auto

using ArgMacros

let  # Global variables can't be typed in Julia < 1.7, so as workaorund we enclose in let block.
    @inlinearguments begin
        @argumentrequired Int imgsize "--size"
        @arghelp "Image size [pixels] e.g. --size 4000"
        @argumentrequired String name "--name"
        @arghelp "Filename prefix used for output files e.g. [prefix]-dirty.fits, [prefix]-restored.fits, etc."
        @argumentrequired Float64 scale "--scale"
        @arghelp "The angular width of a pixel at the origin on the projection [arcsecond]"
        @argumentrequired Symbol weight "--weight"
        @arghelp "The weighting scheme applied to gridded visibilities [uniform | natural | briggs]"
        @argumentdefault Float64 0 briggsweight "--briggs"
        @argumentdefault Int 15 subgridpadding "--subgridpadding"
        @argumentdefault Int 96 subgridsize "--subgridsize"
        @argumentdefault Float64 0.8 mgain "--mgain"
        @argumentdefault Bool true gpu "--gpu"
        @argumentdefault Int 10 miter "--miter"
        @argumentdefault Float64 3 autothreshold "--autothreshold"
        @argumentdefault Int 200 wstep "--wstep"
        @argumentdefault Float64 1e-3 imgthreshold "--imgthreshold"
        @argumentdefault Float64 1e-6 taperthreshold "--taperthreshold"
        @argumentdefault Int 0 chanstart "--chanstart"
        @argumentdefault Int 0 chanstop "--chanstop"
        @positionalrequired String msetname "mset"
        @arghelp "Path to measurement set"
    end

    if weight ∉ (:uniform, :natural, :briggs)
        println(stderr, "--weight $(weight) is not valid. Options: uniform | natural | briggs")
    end

    using Pigi

    # Set wrapper and precision based on CPU v GPU precessing
    wrapper = gpu ? Pigi.CuArray : Array
    precision = gpu ? Float32 : Float64

    Pigi.main(;
        imgsize, subgridpadding, mgain, name, wrapper, miter, autothreshold, wstep,
        imgthreshold, taperthreshold, msetname, scale, chanstart, chanstop,
        subgridsize, precision, weight, briggsweight
    )
end