using AMDGPU
using CUDA
using CUDAKernels
using DSP: conv
using FFTW
using KernelAbstractions
using Pigi
using PyCall
using PyPlot: PyPlot as plt
using Random
using ROCKernels
using StaticArrays
using StructArrays
using Test
using Unitful
using UnitfulAngles

if has_cuda_gpu()
    GPUArray = CuArray
elseif has_rocm_gpu()
    GPUArray = ROCArray
else
    error("No known GPU available for tests")
end

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