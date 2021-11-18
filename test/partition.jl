@testset "Partition" begin
    # Create set of uvdatum with random u, v, coordinates and fixed w
    uvdata = Pigi.UVDatum{Float64}[]
    for (u, v) in zip(rand(Float64, 100000), rand(Float64, 100000))
        u = u * 100 - 50
        v = v * 100 - 50
        push!(uvdata, Pigi.UVDatum{Float64}(0, 0, u, v, 0, [0 0; 0 0], [0 0; 0 0]))
    end

    # Set special seed UVDatum at start
    uvdata[1].u = 20
    uvdata[1].v = 20

    gridspec = Pigi.GridSpec(100, 100, scaleuv=1)
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=1)
    padding = 8
    wstep = 1

    subgrids = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep)
    subgrid = subgrids[1]

    @test subgrid.u0px == 71
    @test subgrid.v0px == 71

    us = [uvdatum.u for uvdatum in subgrid.data]
    vs = [uvdatum.v for uvdatum in subgrid.data]

    @test 23.9 < maximum(us .- 19.5) <= 24
    @test -23.9 > minimum(us .- 19.5) > -24
    @test 23.9 < maximum(vs .- 19.5) <= 24
    @test -23.9 > minimum(vs .- 19.5) > -24
end