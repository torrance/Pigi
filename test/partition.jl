@testset "Partition" begin
    # Create set of uvdatum with random u, v, coordinates and fixed w
    uvdata = Pigi.UVDatum{Float64}[]
    for (u, v) in zip(rand(Float64, 100000), rand(Float64, 100000))
        u = u * 100 - 50
        v = v * 100 - 50
        push!(uvdata, Pigi.UVDatum{Float64}(0, 0, u, v, 0, [0 0; 0 0], [0 0; 0 0]))
    end

    # Set special seed UVDatum at start
    uvdata[1] = Pigi.UVDatum{Float64}(0, 0, 20, 20, 0, [0 0; 0 0], [0 0; 0 0])

    gridspec = Pigi.GridSpec(100, 100, scaleuv=1)
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=1)
    padding = 8
    wstep = 1

    subgrids = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, (l, m) -> 1)
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

@testset "Add subgrid" begin
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=1)
    Aleft = Aright = ones(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    subgrid = Pigi.Subgrid(346, 346, 0., 0., 0., subgridspec, Aleft, Aright, Pigi.UVDatum{Float64}[])

    expected = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    grid = rand(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    expected[346 - 32:346 + 31, 346 - 32:346 + 31] .= grid

    master = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    Pigi.addsubgrid!(master, grid, subgrid)

    @test all(x -> x[1] == x[2], zip(master, expected))

    # Negative grid
    subgrid = Pigi.Subgrid(4, 346, 0., 0., 0., subgridspec, Aleft, Aright, Pigi.UVDatum{Float64}[])

    expected = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    grid = rand(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    expected[1:4 + 31, 346 - 32:346 + 31] .= grid[30:end, :]

    master = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    Pigi.addsubgrid!(master, grid, subgrid)

    @test all(master .== expected)
end