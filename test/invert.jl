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
    expected = reinterpret(reshape, Complex{precision}, expected)

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

    workunits = Pigi.partition(uvdata, paddedgridspec, subgridspec, padding, wstep)
    Pigi.applyweights!(workunits, weighter)
    img = Pigi.invert(workunits, paddedgridspec, taper, subtaper, wrapper)

    img = img[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]

    img = reinterpret(reshape, Complex{precision}, img)

    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < atol

    # vmin, vmax = extrema(real, expected[1, :, :])
    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(img[1, :, :]); vmin, vmax)
    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(expected[1, :, :]); vmin, vmax)
    # plt.subplot(1, 3, 3)
    # diff = real.(img[1, :, :] .- expected[1, :, :])
    # vmin, vmax = extrema(x -> isfinite(x) ? x : 0, diff)
    # plt.imshow(diff; vmin, vmax)
    # plt.colorbar()
    # plt.show()
end