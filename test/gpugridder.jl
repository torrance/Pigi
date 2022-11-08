@testset "GPU Gridding with multiple subgrids" for (precision, atol) in [(Float32, 1e-5), (Float64, 1e-8)]
    gridspec = Pigi.GridSpec(1200, 1200, scaleuv=1)
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=1)

    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    Aterms = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for _ in 1:100_000
        push!(uvdata, Pigi.UVDatum{precision}(
            0, 0, 500 * (rand(precision) - 1//2), 500 * (rand(precision) - 1//2), precision(0),
            SMatrix{2, 2, precision, 4}(1, 1, 1, 1), rand(SMatrix{2, 2, Complex{precision}, 4})
        ))
    end

    workunits = Pigi.partition(uvdata, gridspec, subgridspec, 15, 1, Aterms)
    @test length(workunits) > 1
    @test all(w -> w.w0 == 0, workunits)  # These should all be one layer

    cpusubgrid = zeros(Pigi.StokesI{precision}, 96, 96)
    gpusubgrid = GPUArray(cpusubgrid)

    Pigi.gridder!(cpusubgrid, workunits, subtaper; makepsf=true)
    Pigi.gridder!(gpusubgrid, workunits, GPUArray(subtaper); makepsf=true)

    @test all(x -> all(isfinite, x), cpusubgrid)
    @test all(x -> all(isfinite, x), gpusubgrid)
    @test maximum(x -> sum(abs, x[1] - x[2]), zip(Array(gpusubgrid), cpusubgrid)) < atol

    cpusubgrid = zeros(Pigi.StokesI{precision}, 96, 96)
    gpusubgrid = GPUArray(cpusubgrid)

    Pigi.gridder!(cpusubgrid, workunits, subtaper)
    Pigi.gridder!(gpusubgrid, workunits, GPUArray(subtaper))
    @test maximum(x -> sum(abs, x[1] - x[2]), zip(Array(gpusubgrid), cpusubgrid)) < atol

    # gpusubgrid = [real(x[1]) for x in Array(gpusubgrid)]
    # cpusubgrid = [real(x[1]) for x in cpusubgrid]
    # plt.subplot(1, 2, 1)
    # plt.imshow(real.(gpusubgrid))
    # plt.subplot(1, 2, 2)
    # plt.imshow(real.(cpusubgrid))
    # plt.show()
end