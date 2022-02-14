using BenchmarkTools
using CUDA
using Pigi
using Profile
using InteractiveUtils: @code_warntype
using StaticArrays

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
=#
begin
    precision = Float64

    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)
    padding = 14

    visgrid = zeros(Complex{precision}, 4, 128, 128)
    visgrid[:, 1 + padding:end - padding, 1 + padding:end - padding] = rand(Complex{precision}, 4, 128 - 2 * padding, 128 - 2 * padding)

    uvdata = Pigi.UVDatum{precision}[]
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

    b = @benchmark Pigi.gridder($workunit) evals=1 samples=10 seconds=60
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
=#
begin
    precision = Float64
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1)

    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    uvdata = Pigi.UVDatum{precision}[]
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
    b = @benchmark Pigi.degridder!($workunit, $grid, Pigi.degridop_replace) evals=1 samples=10 seconds=60
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
    Pigi.gpugridder(workunits[1])
    println("Done.")

    b = @benchmark begin
        gridded = 0
        CUDA.@profile CUDA.NVTX.@range "Gridding" Base.@sync for workunit in $workunits
            Base.@async begin
                Pigi.gpugridder(workunit)
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
    Pigi.gpudegridder!(workunits[1], visgrid, Pigi.degridop_replace)
    println("Done.")

    b = @benchmark begin
        CUDA.@profile CUDA.NVTX.@range "Degridding" Base.@sync for (i, workunit) in enumerate($workunits)
            Base.@async Pigi.gpudegridder!(workunit, visgrid, Pigi.degridop_replace)
            print("\rDegridded $(i)/$(length($workunits))")
        end
        println("")
    end evals=1 samples=5 seconds=120
    show(stdout, MIME"text/plain"(), b)
    println()
end