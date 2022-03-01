@testset "GPU Gridding" for (precision, atol) in [(Float32, 1e-5), (Float64, 1e-8)]
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=1)

    taper = Pigi.mkkbtaper(subgridspec)
    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)
    lms = fftfreq(subgridspec.Nx, 1 / subgridspec.scaleuv)
    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        Aleft[lpx, mpx] *= sqrt(taper(l, m))
    end

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for u in -20:20, v in -20:20
        push!(uvdata, Pigi.UVDatum{precision}(
            0, 0, precision(u), precision(v), precision(0), SMatrix{2, 2, precision, 4}(1, 1, 1, 1), rand(SMatrix{2, 2, Complex{precision}, 4})
        ))
    end

    workunit = Pigi.WorkUnit{precision}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    gpusubgrid = Pigi.gridder(workunit, CuArray, makepsf=true)
    cpusubgrid = Pigi.gridder(workunit, Array, makepsf=true)
    @test all(isapprox(x, y; atol) for (x, y) in zip(gpusubgrid, cpusubgrid))

    gpusubgrid = Pigi.gridder(workunit, CuArray)
    cpusubgrid = Pigi.gridder(workunit, Array)
    @test all(isapprox(x, y; atol) for (x, y) in zip(gpusubgrid, cpusubgrid))

    # gpusubgrid = [real(x[1]) for x in gpusubgrid]
    # cpusubgrid = [real(x[1]) for x in cpusubgrid]
    # plt.subplot(1, 2, 1)
    # plt.imshow(real.(gpusubgrid))
    # plt.subplot(1, 2, 2)
    # plt.imshow(real.(cpusubgrid))
    # plt.show()
end