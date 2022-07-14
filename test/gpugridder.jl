@testset "GPU Gridding" for (precision, atol) in [(Float32, 1e-5), (Float64, 1e-8)]
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=1)

    subtaper = Pigi.mkkbtaper(subgridspec, precision)
    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for u in -20:20, v in -20:20
        push!(uvdata, Pigi.UVDatum{precision}(
            0, 0, precision(u), precision(v), precision(0), SMatrix{2, 2, precision, 4}(1, 1, 1, 1), rand(SMatrix{2, 2, Complex{precision}, 4})
        ))
    end

    workunit = Pigi.WorkUnit{precision}(
        49, 49, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    cpusubgrid = zeros(Pigi.LinearData{precision}, 96, 96)
    gpusubgrid = CuArray(cpusubgrid)

    Pigi.gridder!(cpusubgrid, [workunit], subtaper; makepsf=true)
    Pigi.gridder!(gpusubgrid, [workunit], CuArray(subtaper); makepsf=true)

    @test all(x -> all(isfinite, x), cpusubgrid)
    @test all(x -> all(isfinite, x), CuArray(gpusubgrid))
    @test maximum(x -> sum(abs, x[1] - x[2]), zip(Array(gpusubgrid), cpusubgrid)) < atol

    cpusubgrid = zeros(Pigi.LinearData{precision}, 96, 96)
    gpusubgrid = CuArray(cpusubgrid)

    Pigi.gridder!(cpusubgrid, [workunit], subtaper)
    Pigi.gridder!(gpusubgrid, [workunit], CuArray(subtaper))
    @test maximum(x -> sum(abs, x[1] - x[2]), zip(Array(gpusubgrid), cpusubgrid)) < atol

    # gpusubgrid = [real(x[1]) for x in Array(gpusubgrid)]
    # cpusubgrid = [real(x[1]) for x in cpusubgrid]
    # plt.subplot(1, 2, 1)
    # plt.imshow(real.(gpusubgrid))
    # plt.subplot(1, 2, 2)
    # plt.imshow(real.(cpusubgrid))
    # plt.show()
end