@testset "Prediction: $(wrapper) with precision $(precision)" for wrapper in [Array, CuArray], (precision, atol) in [(Float32, 1e-5), (Float64, 1e-6)]
    gridspec = Pigi.GridSpec(2000, 2000, scaleuv=1)

    # Create template skymap (Jy / px) and associated GridSpec
    skymap = zeros(Pigi.StokesI{precision}, gridspec.Nx, gridspec.Ny)
    for (x, y) in zip(rand(700:1300, 1000), rand(700:1300, 1000))
        skymap[x, y] = Pigi.StokesI{precision}(1)
    end
    skymap[501, 501] = Pigi.StokesI{precision}(1)

    # Create uvw sample points
    uvws = rand(3, 5000) .* [1000 1000 500;]' .- [500 500 250;]'
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(uvws)
        push!(uvdata, Pigi.UVDatum{precision}(0, 0, u, v, w, [1 1; 1 1], [0 0; 0 0]))
    end

    # Predict using IDG
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)
    padding = 15
    Aterms = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, 25, Aterms)
    @time Pigi.predict!(workunits, skymap, gridspec, taper, subtaper, wrapper)

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for workunit in workunits
        append!(uvdata, workunit.data)
    end

    # Predict using direct FT
    uvdatadft = deepcopy(uvdata)
    @time dft!(uvdatadft, skymap, gridspec)

    @test maximum(sum(abs.(x[1].data - x[2].data)) for x in zip(uvdata, uvdatadft)) < atol

    # # Plot images
    # expected = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    # idft!(expected, uvdatadft, gridspec, precision(length(uvdatadft)))

    # img = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    # idft!(img, uvdata, gridspec, precision(length(uvdata)))

    # img = reinterpret(reshape, Complex{precision}, img)
    # expected = reinterpret(reshape, Complex{precision}, expected)

    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(img[1, :, :]))
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(expected[1, :, :]))
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.imshow(real.(img[1, :, :] .- expected[1, :, :]))
    # plt.colorbar()

    # plt.show()
end

@testset "Prediction with Aterms" begin
    precision = Float64
    Nbaselines = 10000
    gridspec = Pigi.GridSpec(2000, 2000, scalelm=deg2rad(1/60))
    subgridspec = Pigi.GridSpec(128, 128, scaleuv=gridspec.scaleuv)

    # Set up original sky map
    skymap = zeros(Pigi.StokesI{precision}, 2000, 2000)
    for coord in rand(CartesianIndices((500:1500, 500:1500)), 10)
        skymap[coord] = Pigi.StokesI{precision}(rand())
    end

    # Create Aterms
    sigmalm = 400 * gridspec.scalelm
    subAbeam = map(CartesianIndices((-64:63, -64:63))) do xy
        r = hypot(Tuple(xy)...) * subgridspec.scalelm
        θ = π / 8
        return SMatrix{2, 2, Complex{precision}, 4}(cos(θ), sin(θ), -sin(θ), cos(θ)) * sqrt(exp(-r^2 / (2 * sigmalm^2)))
    end
    Abeam = map(CartesianIndices((-1000:999, -1000:999))) do xy
        r = hypot(Tuple(xy)...) * gridspec.scalelm
        θ = π / 8
        return SMatrix{2, 2, Complex{precision}, 4}(cos(θ), sin(θ), -sin(θ), cos(θ)) * sqrt(exp(-r^2 / (2 * sigmalm^2)))
    end

    # Create UVData by direct FT
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(rand(3, Nbaselines))
        u, v = Pigi.px2lambda(u * 1000 + 500, v * 1000 + 500, gridspec)
        w = 200 * w
        push!(uvdata, Pigi.UVDatum{precision}(1, 1, u, v, w, (1, 1, 1, 1) ./ Nbaselines, (0, 0, 0, 0)))
    end
    uvdata = StructArray(uvdata)

    # Predict using IDG
    wstep = 50
    padding = 15
    subtaper = Pigi.kaiserbessel(subgridspec, precision)
    taper = Pigi.resample(subtaper, subgridspec, gridspec)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, wstep, subAbeam)
    Pigi.predict!(workunits, skymap, gridspec, taper, subtaper, CuArray)

    uvdata = StructVector{Pigi.UVDatum{precision}}(undef, 0)
    for workunit in workunits
        append!(uvdata, workunit.data)
    end

    # Calcuate expected by direct FT
    expected = deepcopy(uvdata)

    # Normalize Abeam for Stokes I and apply to sky map
    skymap = map(Abeam, skymap) do J, val
        J2 = J * J'
        norm = (real(J2[1, 1]) + real(J2[2, 2])) / 2
        return J * convert(Pigi.LinearData{precision}, val) * J'
    end
    dft!(expected, skymap, gridspec)

    @test maximum(sum(abs, x[1].data - x[2].data) for x in zip(uvdata, expected)) < 1e-6

    # # Plot images
    # expectedmap = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    # idft!(expectedmap, expected, gridspec, precision(length(expected)))

    # img = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    # idft!(img, uvdata, gridspec, precision(length(uvdata)))

    # img = Pigi.stokesI(img)
    # expectedmap = Pigi.stokesI(expectedmap)

    # plt.subplot(1, 3, 1)
    # plt.imshow(img)
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.imshow(expectedmap)
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.imshow(img - expectedmap)
    # plt.colorbar()

    # plt.show()
end