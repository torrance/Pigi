@testset "Simple inversion: $(wrapper), $(precision)" for wrapper in [CuArray], (precision, atol) in [(Float32, 1e-4), (Float64, 1e-6)]
    gridspec = Pigi.GridSpec(1000, 1000, scalelm=1u"arcminute")

    padding = 15
    wstep = 50

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
        u, v = Pigi.px2lambda(u * 500 + 250, v * 500 + 250, gridspec)
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
    masterpadding = round(Int, 0.35 * gridspec.Nx)
    paddedgridspec = Pigi.GridSpec(
        gridspec.Nx + 2 * masterpadding,
        gridspec.Ny + 2 * masterpadding,
        scalelm=gridspec.scalelm,
    )
    subgridspec = Pigi.GridSpec(64, 64, scaleuv=paddedgridspec.scaleuv)

    weighter = Pigi.Natural(uvdata, paddedgridspec)

    taper = Pigi.mkkbtaper(gridspec)
    workunits = Pigi.partition(uvdata, paddedgridspec, subgridspec, padding, wstep, taper)
    Pigi.applyweights!(workunits, weighter)
    img = Pigi.invert(workunits, paddedgridspec, taper, wrapper)

    img = img[1 + masterpadding:end - masterpadding, 1 + masterpadding:end - masterpadding]

    img = reinterpret(reshape, Complex{precision}, img)
    @test maximum(isfinite(x) ? abs(x - y) : 0 for (x, y) in zip(img, expected)) < atol

    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(img[1, :, :]))
    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(expected[1, :, :]))
    # plt.subplot(1, 3, 3)
    # plt.imshow(real.(img[1, :, :] .- expected[1, :, :]))
    # plt.colorbar()
    # plt.show()
end