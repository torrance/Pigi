@testset begin "GPU Gridding"
    subgridspec = Pigi.GridSpec(96, 96, scaleuv=1)

    taper = Pigi.mkkbtaper(subgridspec)
    Aleft = Aright = ones(SMatrix{2, 2, ComplexF64, 4}, subgridspec.Nx, subgridspec.Ny)
    lms = fftfreq(subgridspec.Nx, 1 / subgridspec.scaleuv)
    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        Aleft[lpx, mpx] *= sqrt(taper(l, m))
    end

    uvdata = Pigi.UVDatum{Float64}[]
    for u in -20:20, v in -20:20
        push!(uvdata, Pigi.UVDatum(
            0, 0, Float64(u), Float64(v), 0., SMatrix{2, 2, Float64, 4}(1, 1, 1, 1), rand(SMatrix{2, 2, ComplexF64, 4})
        ))
    end

    workunit = Pigi.WorkUnit{Float64}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    gpusubgrid = Pigi.gpugridder(workunit, makepsf=true)
    cpusubgrid = Pigi.gridder(workunit, makepsf=true)
    @test all(isapprox(x, y, atol=1e-5) for (x, y) in zip(gpusubgrid, cpusubgrid))

    gpusubgrid = Pigi.gpugridder(workunit)
    cpusubgrid = Pigi.gridder(workunit)
    @test all(isapprox(x, y, atol=1e-5) for (x, y) in zip(gpusubgrid, cpusubgrid))

    # gpusubgrid = [real(x[1]) for x in gpusubgrid]
    # cpusubgrid = [real(x[1]) for x in cpusubgrid]
    # plt.subplot(1, 2, 1)
    # plt.imshow(real.(gpusubgrid))
    # plt.subplot(1, 2, 2)
    # plt.imshow(real.(cpusubgrid))
    # plt.show()
end