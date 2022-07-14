@testset "Simple inversion: $(wrapper), $(precision)" for wrapper in [Array, CuArray], (precision, atol) in [(Float32, 6e-4), (Float64, 6e-4)]
    gridspec = Pigi.GridSpec(2000, 2000, scalelm=0.5u"arcminute")

    padding = 18
    wstep = 100

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
    vimg, vtruncate = 1e-3, 1e-6
    paddingfactor = Pigi.taperpadding(vimg, vtruncate)

    masterpadding = round(Int, (paddingfactor - 1) * gridspec.Nx / 2)
    paddedgridspec = Pigi.GridSpec(
        gridspec.Nx + 2 * masterpadding,
        gridspec.Ny + 2 * masterpadding,
        scalelm=gridspec.scalelm,
    )
    println("Use padding factor $(paddingfactor) -> $(paddedgridspec.Nx)px x $(paddedgridspec.Ny)")

    subgridspec = Pigi.GridSpec(96, 96, scaleuv=paddedgridspec.scaleuv)

    weighter = Pigi.Natural(precision, uvdata, paddedgridspec)

    subtaper = Pigi.mkkbtaper(subgridspec, precision, threshold=vtruncate)
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

@testset "Inversion with beam (precision: $(precision))" for precision in [Float32, Float64]
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
    subtaper = Pigi.mkkbtaper(subgridspec, precision; threshold=0)
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
    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < 5e-4

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