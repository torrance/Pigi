@testset "Prediction" begin
    precision = Float64
    gridspec = Pigi.GridSpec(2000, 2000, scaleuv=1)

    # Create template skymap (Jy / px) and associated GridSpec
    skymap = zeros(SMatrix{2, 2, Complex{precision}, 4}, gridspec.Nx, gridspec.Ny)
    for (x, y) in zip(rand(700:1300, 1000), rand(700:1300, 1000))
        skymap[x, y] = [1 0; 0 1]
    end
    skymap[501, 501] = [1 0; 0 1]

    # Create uvw sample points
    uvws = rand(3, 5000) .* [1000 1000 500;]' .- [500 500 250;]'
    uvdata = Pigi.UVDatum{precision}[]
    for (u, v, w) in eachcol(uvws)
        push!(uvdata, Pigi.UVDatum{precision}(0, 0, u, v, w, [1 1; 1 1], [0 0; 0 0]))
    end

    # Predict using IDG
    taper = Pigi.mkkbtaper(gridspec)
    padding = 15
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=gridspec.scaleuv)
    workunits = Pigi.partition(uvdata, gridspec, subgridspec, padding, 25, taper)
    @time Pigi.predict!(workunits, skymap, gridspec, taper)

    uvdata = Pigi.UVDatum{precision}[]
    for workunit in workunits
        append!(uvdata, workunit.data)
    end

    # Predict using direct FT
    uvdatadft = deepcopy(uvdata)
    @time dft!(uvdatadft, skymap, gridspec)

    @test all(x -> sum(abs.(x[1].data - x[2].data)) < 1e-7, zip(uvdata, uvdatadft))

    # # Plot images
    # expected = similar(skymap)
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