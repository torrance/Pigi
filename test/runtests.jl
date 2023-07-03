using DSP: conv
using FFTW
using KernelAbstractions
using Pigi
using PyCall
using PyPlot: PyPlot as plt
using Random
using StaticArrays
using StructArrays
using Test
using Unitful
using UnitfulAngles

const GPUArray = Array
println(stderr, "No GPU available for tests; defaulting to CPU")

include("functions.jl")

@testset begin
    include("uvdatum.jl")
    include("measurementset.jl")
    include("partition.jl")
    include("gridspec.jl")
    include("gridder.jl")
    include("weights.jl")
    include("invert.jl")
    include("predict.jl")
    include("clean.jl")
    include("psf.jl")
    include("gpugridder.jl")
    include("gpudegridder.jl")
    include("utility.jl")
    include("mwabeam.jl")
    include("coordinates.jl")
end