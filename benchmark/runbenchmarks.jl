using BenchmarkTools
using CUDA
using DSP: conv
using InteractiveUtils: @code_warntype
using Pigi
using Profile
using StaticArrays
using StructArrays

println("Running benchmarks...")

#=
2021/11/09 : Nimbus
    Time (mean ± σ): 15.936 s ± 134.312 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 9.06 MiB, allocs estimate: 86217
2022/02/07 Nimbus
    Time (mean ± σ): 15.375 s ± 279.428 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 12.41 MiB, allocs estimate: 85526
    Note: upgrade to Julia 1.7.1
=#
begin
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=192)
    b = @benchmark sum(1 for uvdatum in Pigi.read($mset)) evals=1 samples=3 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2021/11/09 : Nimbus
    Time (mean ± σ): 12.186 s ± 2.400 s GC (mean ± σ): 6.05% ± 2.37%
    Memory estimate: 19.24 GiB, allocs estimate: 27066
2021/02/07 : Nimbus
    Time (mean ± σ): 12.762 s ± 2.191 s GC (mean ± σ): 10.07% ± 3.68%
    Memory estimate: 22.30 GiB, allocs estimate: 23394
    Note: upgrade to Julia 1.7.1
=#
begin
    println("Reading mset...")
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=384)
    println("Mset opened...")
    uvdata = collect(Pigi.read(mset))
    println(typeof(uvdata))
    println("Done.")

    scalelm = sin(deg2rad(15 / 3600))
    gridspec = Pigi.GridSpec(4000, 4000, scalelm=scalelm)
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    padding = 8
    wstep = 10

    b = @benchmark Pigi.partition($uvdata, $gridspec, $subgridspec, $padding, $wstep, (l, m) -> 1) evals=1 samples=5 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2021/11/09 : Nimbus
    Time (mean ± σ): 4.568 s ± 16.015 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 2.00 MiB, allocs estimate: 54
2021/11/25 : Nimbus
    Time (mean ± σ): 4.271 s ± 14.215 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 2.00 MiB, allocs estimate: 27
    Note: enabled @SIMD
2022/02/07 : Nimbus
    Time (mean ± σ): 4.483 s ± 7.480 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 2.00 MiB, allocs estimate: 33
    Note: upgrade to Julia 1.7.1
2022/03/01 : Nimbus
    Time (mean ± σ): 4.860 s ± 4.979 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 2.00 MiB, allocs estimate: 37
    Note: switch the StructArrays
=#
begin
    precision = Float64

    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)
    padding = 14

    visgrid = zeros(Complex{precision}, 4, 128, 128)
    visgrid[:, 1 + padding:end - padding, 1 + padding:end - padding] = rand(Complex{precision}, 4, 128 - 2 * padding, 128 - 2 * padding)

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for vpx in axes(visgrid, 3), upx in axes(visgrid, 2)
        val = visgrid[:, upx, vpx]
        if !all(val .== 0)
            u, v = Pigi.px2lambda(upx, vpx, subgridspec)
            push!(uvdata, Pigi.UVDatum{precision}(
                0, 0, u, v, 0, [1 1; 1 1], val
            ))
        end
    end
    println("Gridding $(length(uvdata)) uvdatum")

    Aleft = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    subtaper = Pigi.mkkbtaper(subgridspec, precision)

    workunits = [Pigi.WorkUnit{precision}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )]
    mastergrid = zeros(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    b = @benchmark Pigi.gridder!(mastergrid, $workunits, subtaper) evals=1 samples=10 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2021/11/30 : Nimbus
    Time (mean ± σ): 5.601 s ± 18.682 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 1.00 MiB, allocs estimate: 29
2022/02/07 : Nimbus
    Time (mean ± σ): 5.664 s ± 10.579 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 1.00 MiB, allocs estimate: 37
    Note: upgrade to Julia 1.7.1
2022/03/01 : Nimbus
    Time (mean ± σ): 5.929 s ± 4.686 ms GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 4.04 MiB, allocs estimate: 49016
    Note: switch to StructArray
=#
begin
    precision = Float64
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1)

    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    uvdata = StructArray{Pigi.UVDatum{precision}}(undef, 0)
    for (upx, vpx) in eachcol(rand(2, 10000))
        upx, vpx = upx * 100 + 14, vpx * 100 + 14
        u, v = Pigi.px2lambda(upx, vpx, subgridspec)
        push!(uvdata, Pigi.UVDatum{precision}(
            0, 0, u, v, 0, [1 1; 1 1], [0 0; 0 0]
        ))
    end
    println("Degridding $(length(uvdata)) uvdatum")

    workunit = Pigi.WorkUnit{precision}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    grid = rand(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    b = @benchmark Pigi.degridder!($workunit, $grid, Pigi.degridop_replace, Array) evals=1 samples=10 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
Function: invert()

2022/03/16 : Nibmus
    Time (mean ± σ): 35.156 s ±  1.220 s GC (mean ± σ): 0.73% ± 0.38%
    Memory estimate: 6.63 GiB, allocs estimate: 81407455
    Note: multithreaded
2022/03/23 : Nimbus
    Time (mean ± σ): 28.856 s ± 76.938 ms GC (mean ± σ): 3.58% ± 0.16%
    Memory estimate: 6.08 GiB, allocs estimate: 81594187
    Note: remove intermdiate data transfers from host <-> device
2022/07/14 : Nimbus
    Time (mean ± σ): 9.548 s ± 171.065 ms GC (mean ± σ): 0.45% ± 0.22%
    Memory estimate: 1.24 GiB, allocs estimate: 540758
    Note: use StokesI <: OutputType; precalculate taper; force π to use precision
=#
begin
    println("Reading mset...")
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=196)
    println("Mset opened...")
    uvdata = StructArray(Pigi.read(mset, precision=Float32))
    println(typeof(uvdata))
    println("Done.")

    scalelm = sin(deg2rad(15 / 3600))
    gridspec = Pigi.GridSpec(9000, 9000, scalelm=scalelm)
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=gridspec.scaleuv)
    padding = 15
    wstep = 200

    println("Partitioning data...")
    subtaper = Pigi.mkkbtaper(subgridspec, Float32)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)
    Aterms = zeros(Pigi.Comp2x2{Float32}, 96, 96)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, Aterms)
    println("Done.")

    println("Compiling...")
    Pigi.invert(Pigi.StokesI{Float32}, workunits[1:1], gridspec, taper, subtaper, CuArray)
    println("Done.")

    b = @benchmark Pigi.invert(Pigi.StokesI{Float32}, workunits, gridspec, taper, subtaper, CuArray) evals=1 samples=5 seconds=300
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
Function: predict!()

2022/02/11 : Nimbus
    Time (mean ± σ): 30.747 s ± 287.560 ms  GC (mean ± σ): 0.43% ± 0.25%
    Memory estimate: 5.43 GiB, allocs estimate: 447389
2022/03/23 : Nimbus
    Time (mean ± σ): 22.880 s ± 339.722 ms GC (mean ± σ): 1.56% ± 0.75%
    Memory estimate: 2.44 GiB, allocs estimate: 518681
    Note: remove intermdiate data transfers from host <-> device
2022/07/15 : Nimbus
    Time (mean ± σ): 19.649 s ± 317.612 ms GC (mean ± σ): 1.63% ± 0.79%
    Memory estimate: 2.44 GiB, allocs estimate: 479952
    Note: get benchmarks working again; unknown changes
2022/07/15 : Nimbus
    Time (mean ± σ): 8.924 s ± 264.739 ms GC (mean ± σ): 0.46% ± 0.25%
    Memory estimate: 644.92 MiB, allocs estimate: 435547
    Note: use StokesI, ensure consistent precision throughout
=#
begin
    println("Reading mset...")
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=196)
    println("Mset opened...")
    uvdata = collect(Pigi.read(mset, precision=Float32))
    println(typeof(uvdata))
    println("Done.")

    scalelm = sin(deg2rad(15 / 3600))
    gridspec = Pigi.GridSpec(9000, 9000, scalelm=scalelm)
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=gridspec.scaleuv)
    padding = 15
    wstep = 200

    println("Partitioning data...")
    subtaper = Pigi.mkkbtaper(subgridspec, Float32)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)
    Aterms = ones(Pigi.Comp2x2{Float32}, 96, 96)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, Aterms)
    println("Done.")

    # Create random sky grid
    grid = zeros(Pigi.StokesI{Float32}, 9000, 9000)
    grid[rand(1:9000 * 9000, 6000)] .= rand(Pigi.StokesI{Float32}, 6000)

    # Compile CUDA kernel
    println("Compiling CUDA kernel...")
    Pigi.predict!(workunits[1:1], grid, gridspec, taper, subtaper, CuArray; degridop=Pigi.degridop_replace)
    println("Done.")

    b = @benchmark begin
        Pigi.predict!(workunits, grid, gridspec, taper, subtaper, CuArray; degridop=Pigi.degridop_replace)
    end evals=1 samples=5 seconds=300
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2022/02/24 : Nimbus
    Time (mean ± σ): 25.431 s ± 24.692 ms GC (mean ± σ): 0.03% ± 0.06%
    Memory estimate: 488.29 MiB, allocs estimate: 86
2022/02/24 : Nimbus
    Time (mean ± σ): 4.510 s ± 44.600 ms GC (mean ± σ): 0.18% ± 0.36%
    Memory estimate: 488.65 MiB, allocs estimate: 6391
    Note: initial GPU implementation of findabsmax() and subtactpsf()
2022/02/24 : Nimbus
    Time (mean ± σ): 867.602 ms ± 25.458 ms GC (mean ± σ): 0.93% ± 1.79%
    Memory estimate: 560.19 MiB, allocs estimate: 6791
    Note: multi-block reduction
2022/02/24 : Nimbus
    Time (mean ± σ): 3.618 s ± 56.215 ms GC (mean ± σ): 0.42% ± 0.28%
    Memory estimate: 1.18 GiB, allocs estimate: 69001
    Note: increased benchmark niter=100 -> niter=1000
2022/02/25 : Nimbus
    Time (mean ± σ): 2.671 s ± 38.050 ms GC (mean ± σ): 0.34% ± 0.66%
    Memory estimate: 492.54 MiB, allocs estimate: 77278
    Note: block reduction kernel, and more efficient thread-level reduction
2022/02/25 : Nimbus
    Time (mean ± σ): 1.820 s ± 41.545 ms  GC (mean ± σ): 0.66% ± 1.18%
    Memory estimate: 497.35 MiB, allocs estimate: 169040
    Note: mapreduce on findabsmax()
=#
begin
    expectedcomponentmap = zeros(Float64, 8000, 8000)

    for (xpx, ypx) in eachcol(rand(1:8000, 2, 1000))
        expectedcomponentmap[xpx, ypx] += rand()
    end

    psf = map(CartesianIndices((-32:31, -32:31))) do idx
        sigmax = 5
        sigmay = 10
        x, y = Tuple(idx)
        return exp(-x^2 / (2 * sigmax^2) - y^2 / (2 * sigmay^2))
    end

    img = conv(expectedcomponentmap, psf)[1 + 32:end - 31, 1 + 32:end - 31]

    b = @benchmark begin
        componentmap, iter = Pigi.clean!($img, $psf, mgain=1, threshold=0, niter=1000)
    end evals=1 samples=5 seconds=180
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2022/02/25 : Nimbus
    Time (mean ± σ): 3.174 ms ± 153.783 μs GC (mean ± σ): 1.01% ± 0.35%
    Memory estimate: 827.63 KiB, allocs estimate: 65
2022/02/25 : Nimbus
    Time (mean ± σ): 2.302 ms ± 57.978 μs GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 4.05 KiB, allocs estimate: 73
    Note: block reduction kernel, and more efficient thread-level reduction
2022/02/25 : Nimbus
    Time (mean ± σ): 1.239 ms ± 83.558 μs GC (mean ± σ): 0.00% ± 0.00%
    Memory estimate: 8.02 KiB, allocs estimate: 134
    Note: just use mapreduce :/
=#
begin
    arr = CUDA.rand(9000, 9000)

    b = @benchmark begin
        CUDA.findmax($arr)
    end evals=100 samples=10 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()

    b = @benchmark begin
        Pigi.findabsmax($arr)
    end evals=100 samples=10 seconds=60
    show(stdout, MIME"text/plain"(), b)
    println()
end