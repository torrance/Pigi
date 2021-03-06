@testset "Simple inversion: $(wrapper), $(precision)" for wrapper in [Array, CuArray], (precision, atol) in [(Float32, 5e-5), (Float64, 1e-9)]
    gridspec = Pigi.GridSpec(2000, 2000, scalelm=0.5u"arcminute")

    padding = 18
    wstep = 25

    # Source locations in radians, wrt to phase center, with x degree FOV
    sources = deg2rad.(
        rand(Float64, 2, 500) * 30 .- 15
    )
    sources[:, 1] .= 0

    println("Predicting UVDatum...")
    N = 20000
    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, N)
    Threads.@threads for i in 1:N
        u, v, w = rand(Float64, 3)
        u, v = Pigi.px2lambda(u * 500 + gridspec.Nx ÷ 2 - 250, v * 500 + gridspec.Ny ÷ 2 - 250, gridspec)
        w = w * 500

        val = zero(SMatrix{2, 2, Complex{precision}, 4})
        for (ra, dec) in eachcol(sources)
            l, m = sin(ra), sin(dec)
            val += @SMatrix([1 0; 0 1]) * exp(-2π * 1im * (u * l + v * m + w * Pigi.ndash(l, m)))
        end
        uvdata[i] = Pigi.UVDatum{precision}(0, 0, u, v, w, @SMatrix(precision[1 1; 1 1]), val)
    end
    println("Done.")

    # Calculate expected output
    expected = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    @time idft!(expected, uvdata, gridspec, precision(length(uvdata)))
    expected = map(expected) do x
        return (x[1, 1] + x[2, 2]) / 2
    end

    # IDG
    paddingfactor = 1.5
    masterpadding = round(Int, (paddingfactor - 1) * gridspec.Nx / 2)
    paddedgridspec = Pigi.GridSpec(
        gridspec.Nx + 2 * masterpadding,
        gridspec.Ny + 2 * masterpadding,
        scalelm=gridspec.scalelm,
    )
    println("Use padding factor $(paddingfactor) -> $(paddedgridspec.Nx)px x $(paddedgridspec.Ny)")

    subgridspec = Pigi.GridSpec(96, 96, scaleuv=paddedgridspec.scaleuv)

    weighter = Pigi.Natural(precision, uvdata, paddedgridspec)

    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    taper = Pigi.resample(subtaper, subgridspec, paddedgridspec)
    Aterms = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)

    workunits = Pigi.partition(uvdata, paddedgridspec, subgridspec, padding, wstep, Aterms)
    Pigi.applyweights!(workunits, weighter)
    img = Pigi.invert(Pigi.StokesI{precision}, workunits, paddedgridspec, taper, subtaper, wrapper)

    img = img[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]

    img = reinterpret(Complex{precision}, img)

    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < atol

    # vmin, vmax = extrema(real, expected)
    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(img); vmin, vmax)
    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(expected); vmin, vmax)
    # plt.subplot(1, 3, 3)
    # diff = real.(img .- expected)
    # vmin, vmax = extrema(x -> isfinite(x) ? x : 0, diff)
    # plt.imshow(diff; vmin, vmax)
    # plt.colorbar()
    # plt.show()
end

@testset "Inversion with beam (precision: $(precision))" for (precision, atol) in [(Float32, 5e-4), (Float64, 5e-5)]
    Random.seed!(123456)
    Nbaselines = 10000
    gridspec = Pigi.GridSpec(2000, 2000, scalelm=deg2rad(2/60))
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=gridspec.scaleuv)

    # Set up original sky map
    skymap = zeros(SMatrix{2, 2, Complex{precision}, 4}, 2000, 2000)
    for coord in rand(CartesianIndices((500:1500, 500:1500)), 100)
        skymap[coord] = rand() * one(SMatrix{2, 2, ComplexF64, 4})
    end

    # Create Aterms
    beam = Pigi.MWABeam(zeros(Int, 16))
    coords_radec = Pigi.grid_to_radec(subgridspec, (0., π / 2))
    coords_altaz = reverse.(coords_radec)
    subAbeam = Pigi.getresponse(beam, coords_altaz, 150e6)

    Abeam = Pigi.resample(subAbeam, subgridspec, gridspec)

    # Apply Aterms to skymap
    map!(skymap, Abeam, skymap) do J, val
        return J * val * J'
    end

    # Create UVData by direct FT
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(randn(3, Nbaselines))
        u, v = Pigi.px2lambda(u * 100 + 1000, v * 100 + 1000, gridspec)
        w = 200 * w
        push!(uvdata, Pigi.UVDatum{precision}(1, 1, u, v, w, (1, 1, 1, 1) ./ Nbaselines, (0, 0, 0, 0)))
    end
    uvdata = StructArray(uvdata)
    dft!(uvdata, skymap, gridspec)

    expected = zeros(SMatrix{2, 2, Complex{precision}, 4}, 2000, 2000)
    idft!(expected, uvdata, gridspec, precision(length(uvdata)))
    map!(expected, expected, Abeam) do val, J
        invJ = inv(J)
        return invJ * val * invJ'
    end

    power = map(Abeam) do J
        return real(sum((J * J')[[1, 4]])) / 2
    end

    # IDG
    wstep = 25
    padding = 16
    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)
    subAbeam = convert(Array{Pigi.Comp2x2{precision}}, subAbeam)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, subAbeam)
    println("N workunits: ", length(workunits))
    img = Pigi.invert(Pigi.StokesI{precision}, workunits, gridspec, taper, subtaper, CuArray)

    expected = map((expected .* power)[500:1500, 500:1500]) do x
        real(x[1, 1] + x[2, 2]) / 2
    end
    img = real.(img[500:1500, 500:1500])

    println("Max error: ", maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)))
    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < atol

    # diff = img - expected
    # plt.subplot(1, 3, 1)
    # plt.imshow(expected)
    # plt.subplot(1, 3, 2)
    # plt.imshow(img)
    # plt.subplot(1, 3, 3)
    # plt.imshow(diff)
    # plt.colorbar()
    # plt.show()
end

@testset "Inversion with multiple beams (precision: $(precision))" for (precision, atol) in [(Float32, 5e-5), (Float64, 1e-6)]
    Random.seed!(123456)
    Nbaselines = 20000
    imgsize = 1200

    paddingfactor = 1.5
    masterpadding = round(Int, (paddingfactor - 1) * imgsize) ÷ 2
    gridspec = Pigi.GridSpec(imgsize + 2 * masterpadding, imgsize + 2 * masterpadding, scalelm=scalelm=deg2rad(1.8/60))
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=gridspec.scaleuv)

    # Set up original sky map
    skymap = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    offset = CartesianIndex(masterpadding, masterpadding)
    for coord in rand(CartesianIndices((1:imgsize, 1:imgsize)), 100)
        skymap[coord + offset] = one(SMatrix{2, 2, ComplexF64, 4})
    end

    # Create Aterms for two beams
    beam = Pigi.MWABeam(Int[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
    coords_radec = Pigi.grid_to_radec(subgridspec, (0., π / 2))
    coords_altaz = reverse.(coords_radec)
    subAbeam1 = Pigi.getresponse(beam, coords_altaz, 150e6)
    Abeam1 = Pigi.resample(subAbeam1, subgridspec, gridspec)

    skymap1 = map(Abeam1, skymap) do J, val
        return J * val * J'
    end

    beam = Pigi.MWABeam(zeros(Int, 16))
    coords_radec = Pigi.grid_to_radec(subgridspec, (0., π / 2))
    coords_altaz = reverse.(coords_radec)
    subAbeam2 = Pigi.getresponse(beam, coords_altaz, 150e6)
    Abeam2 = Pigi.resample(subAbeam2, subgridspec, gridspec)

    skymap2 = map(Abeam2, skymap) do J, val
        return J * val * J'
    end

    # Create UVData by direct FT
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(randn(3, Nbaselines))
        u, v = Pigi.px2lambda(u * 100 + gridspec.Nx / 2, v * 100 + gridspec.Ny / 2, gridspec)
        w = 200 * w
        push!(uvdata, Pigi.UVDatum{precision}(1, 1, u, v, w, (1, 1, 1, 1) ./ Nbaselines, (0, 0, 0, 0)))
    end
    uvdata[1] = Pigi.UVDatum{precision}(1, 1, 0, 0, 0, (1, 1, 1, 1) ./ Nbaselines, (0, 0, 0, 0))
    uvdata = StructArray(uvdata)

    idxs1 = rand(Bool[true, true, false], Nbaselines)
    idxs1 = rand(Bool[true], Nbaselines)
    idxs2 = (!).(idxs1)

    dft!(view(uvdata, idxs1), skymap1, gridspec)
    dft!(view(uvdata, idxs2), skymap2, gridspec)

    # Create expected dirty image by direct FT
    expected1 = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    idft!(expected1, uvdata[idxs1], gridspec, precision(length(uvdata)))
    map!(expected1, expected1, Abeam1) do val, J
        invJ = inv(J)
        val *= sum(real, (J * J')[[1, 4]]) / 2
        return invJ * val * invJ'
    end

    expected2 = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    idft!(expected2, uvdata[idxs2], gridspec, precision(length(uvdata)))
    map!(expected2, expected2, Abeam2) do val, J
        invJ = inv(J)
        val *= sum(real, (J * J')[[1, 4]]) / 2
        return invJ * val * invJ'
    end

    expected = (expected1 + expected2)[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]
    expected = map(expected) do x
        real(x[1, 1] + x[2, 2]) / 2
    end

    # IDG
    wstep = 25
    padding = 16

    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)

    subAbeam1 = convert(Array{Pigi.Comp2x2{precision}}, subAbeam1)
    workunits1 = Pigi.partition(uvdata[idxs1], gridspec, subgridspec, padding, wstep, subAbeam1)
    subAbeam2 = convert(Array{Pigi.Comp2x2{precision}}, subAbeam2)
    workunits2 = Pigi.partition(uvdata[idxs2], gridspec, subgridspec, padding, wstep, subAbeam2)
    workunits = vcat(workunits1, workunits2)
    println("N workunits: ", length(workunits))

    img = Pigi.invert(Pigi.StokesI{precision}, workunits, gridspec, taper, subtaper, CuArray)
    img = real.(img)[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]

    println("Max error: ", maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)))
    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < 5e-5

    # diff = img - expected
    # plt.subplot(1, 3, 1)
    # plt.imshow(expected)
    # plt.subplot(1, 3, 2)
    # plt.imshow(img; vmin=minimum(expected), vmax=maximum(expected))
    # plt.subplot(1, 3, 3)
    # plt.imshow(diff, vmin=-1e-8, vmax=1e-8)
    # plt.colorbar()
    # plt.show()
end