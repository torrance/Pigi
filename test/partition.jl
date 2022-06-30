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
    Aterms = ones(SMatrix{2, 2, Complex{Float64}, 4}, subgridspec.Nx, subgridspec.Ny)

    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, Aterms)
    workunit = workunits[1]

    @test workunit.u0px == 71
    @test workunit.v0px == 71

    us = [uvdatum.u for uvdatum in workunit.data]
    vs = [uvdatum.v for uvdatum in workunit.data]

    @test 23.9 < maximum(us .- 19.5) <= 24
    @test -23.9 > minimum(us .- 19.5) > -24
    @test 23.9 < maximum(vs .- 19.5) <= 24
    @test -23.9 > minimum(vs .- 19.5) > -24
end

@testset "Add subgrid" for wrapper in [Array, CuArray]
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=1)
    Aleft = Aright = ones(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    workunit = Pigi.WorkUnit(346, 346, 0., 0., 0., subgridspec, Aleft, Aright, StructVector{Pigi.UVDatum{Float64}}(undef, 0))

    expected = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    grid = rand(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    expected[346 - 32:346 + 31, 346 - 32:346 + 31] .= grid

    master = wrapper(zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000))
    Pigi.addsubgrid!(master, wrapper(grid), workunit)

    @test Array(master) == expected

    # Negative grid
    workunit = Pigi.WorkUnit(4, 346, 0., 0., 0., subgridspec, Aleft, Aright, StructVector{Pigi.UVDatum{Float64}}(undef, 0))

    expected = zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)
    grid = rand(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    expected[1:4 + 31, 346 - 32:346 + 31] .= grid[30:end, :]

    master = wrapper(zeros(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000))
    Pigi.addsubgrid!(master, wrapper(grid), workunit)

    @test Array(master) == expected
end

@testset "Extract subgrid" for wrapper in [Array, CuArray]
    mastergrid = rand(SMatrix{2, 2, ComplexF64, 4}, 1000, 1000)

    expected = mastergrid[231 - 32:231 + 31, 785 - 32:785 + 31]

    subgridspec = Pigi.GridSpec(64, 64, scaleuv=1)
    Aleft = Aright = ones(SMatrix{2, 2, ComplexF64, 4}, 64, 64)
    workunit = Pigi.WorkUnit{Float64}(231, 785, 0., 0., 0., subgridspec, Aleft, Aright, StructVector{Pigi.UVDatum{Float64}}(undef, 0))

    @test expected == Array(Pigi.extractsubgrid(wrapper(mastergrid), workunit))
end