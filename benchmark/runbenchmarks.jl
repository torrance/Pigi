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

    taper = Pigi.mkkbtaper(subgridspec)

    workunit = Pigi.WorkUnit{precision}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    b = @benchmark Pigi.gridder($workunit, Array) evals=1 samples=10 seconds=60
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
2022/02/07 : Nimbus
    Time (mean ± σ): 24.062 s ± 1.728 s GC (mean ± σ): 0.05% ± 0.06%
    Memory estimate: 885.26 MiB, allocs estimate: 354185
2022/02/11 : Nimus
    Time (mean ± σ): 12.147 s ± 900.924 ms GC (mean ± σ): 0.07% ± 0.14%
    Memory estimate: 883.59 MiB, allocs estimate: 332030
    Note: subgrid cell per block, kernel performs block-local reduction over uvdata
2022/02/14 : Nimbus
    Time (mean ± σ): 6.608 s ± 725.974 ms GC (mean ± σ): 0.55% ± 0.30%
    Memory estimate: 2.10 GiB, allocs estimate: 488955
    Note: iterate over uvdatum fields as separate arrays; perform fft on CPU
2022/02/28 : Nimbus
    Time (mean ± σ): 5.361 s ± 512.201 ms GC (mean ± σ): 0.45% ± 0.39%
    Memory estimate: 1.85 GiB, allocs estimate: 363213
    Note: one thread per subgrid cell, iterate over uvdata within the thread
2022/02/28 : Nimbus
    Time (mean ± σ): 4.794 s ± 1.011 s GC (mean ± σ): 0.25% ± 0.41%
    Memory estimate: 640.10 MiB, allocs estimate: 434324
    Note: using StructArrays
=#
begin
    println("Reading mset...")
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=96)
    println("Mset opened...")
    uvdata = collect(Pigi.read(mset, precision=Float32))
    println(typeof(uvdata))
    println("Done.")

    scalelm = sin(deg2rad(15 / 3600))
    gridspec = Pigi.GridSpec(4000, 4000, scalelm=scalelm)
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    padding = 8
    wstep = 10

    println("Partitioning data...")
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, (l, m) -> 1)
    println("Done.")

    # Compile CUDA kernel
    println("Compiling CUDA kernel...")
    Pigi.gridder(workunits[1], CuArray)
    println("Done.")

    b = @benchmark begin
        gridded = 0
        CUDA.@profile CUDA.NVTX.@range "Gridding" Base.@sync for workunit in $workunits
            Base.@async begin
                Pigi.gridder(workunit, CuArray)
                gridded += 1
                print("\rGridded $(gridded)/$(length($workunits))")
            end
        end
        println("")
    end evals=1 samples=5 seconds=120
    show(stdout, MIME"text/plain"(), b)
    println()
end

#=
2022/02/11 : Nimbus
    Time (mean ± σ): 4.801 s ± 964.457 ms GC (mean ± σ): 0.13% ± 0.21%
    Memory estimate: 454.40 MiB, allocs estimate: 373964
=#
begin
    println("Reading mset...")
    path = "../testdata/1215555160/1215555160.ms"
    mset = Pigi.MeasurementSet(path, chanstart=1, chanstop=96)
    println("Mset opened...")
    uvdata = collect(Pigi.read(mset, precision=Float32))
    println(typeof(uvdata))
    println("Done.")

    scalelm = sin(deg2rad(15 / 3600))
    gridspec = Pigi.GridSpec(4000, 4000, scalelm=scalelm)
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    padding = 8
    wstep = 10

    println("Partitioning data...")
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, (l, m) -> 1)
    println("Done.")

    # Create random visgrid
    visgrid = rand(SMatrix{2, 2, ComplexF32, 4}, 64, 64)

    # Compile CUDA kernel
    println("Compiling CUDA kernel...")
    Pigi.degridder!(workunits[1], visgrid, Pigi.degridop_replace, CuArray)
    println("Done.")

    b = @benchmark begin
        CUDA.@profile CUDA.NVTX.@range "Degridding" Base.@sync for (i, workunit) in enumerate($workunits)
            Base.@async Pigi.degridder!(workunit, visgrid, Pigi.degridop_replace, CuArray)
            print("\rDegridded $(i)/$(length($workunits))")
        end
        println("")
    end evals=1 samples=5 seconds=120
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