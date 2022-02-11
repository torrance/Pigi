@testset begin
    precision = Float64
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1)

    # Create visgrid
    visgrid = rand(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)

    # Create uvw sample points
    uvws = rand(3, 5000) .* [100 100 20;]' .- [50 50 10;]'
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(uvws)
        push!(uvdata, Pigi.UVDatum{precision}(0, 0, u, v, w, [1 1; 1 1], [0 0; 0 0]))
    end

    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    cpuworkunit = Pigi.WorkUnit(0, 0, 0., 0., 0., subgridspec, Aleft, Aright, uvdata)
    gpuworkunit = deepcopy(cpuworkunit)

    Pigi.gpudegridder!(gpuworkunit, visgrid, Pigi.degridop_replace)
    Pigi.degridder!(cpuworkunit, visgrid, Pigi.degridop_replace)

    @test all(isapprox(x.data, y.data, atol=1e-8) for (x, y) in zip(gpuworkunit.data, cpuworkunit.data))
end