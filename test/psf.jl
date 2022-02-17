@testset "PSF fitting" begin
    # Create a realistic psf with sidelobes
    pa = deg2rad(23)
    sigmax = 100
    sigmay = 150

    psf = map(CartesianIndices((-1000:999, -1000:999))) do idx
        x, y = Tuple(idx)
        x, y = x * cos(pa) - y * sin(pa), x * sin(pa) + y * cos(pa)
        return exp(-x^2 / (2 * sigmax^2) - y^2 / (2 * sigmay^2))
    end

    # Make the psf converage patchy
    map!(psf, psf) do val
        rand() < 0.8 ? 0 : val
    end

    psf = real.(fftshift(fft(ifftshift(psf))))
    psf ./= psf[1001, 1001]

    threshold = 0.05
    psfclipped = Pigi.psfclip(psf, threshold)
    println(size(psfclipped))

    @test size(psfclipped, 1) == size(psfclipped, 2)

    x0, y0 = size(psfclipped) .÷ 2 .+ 1
    @test psfclipped[x0, y0] ≈ 1

    @test all(psf[1001 - x0 + 1:1001 + x0 - 2, 1001 - y0 + 1:1001 + y0 - 2] .== psfclipped)

    psf[1001 - x0 + 1:1001 + x0 - 2, 1001 - y0 + 1:1001 + y0 - 2] .= 0
    @test all(v -> abs(v) < threshold, psf)

    params, fitted = Pigi.psffit(psfclipped)

    @test all(x -> abs(x[1] - x[2]) < 0.012, zip(fitted, psfclipped))

    # plt.subplot(1, 3, 1)
    # plt.imshow(psfclipped)
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.imshow(fitted)
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(psfclipped .- fitted)
    # plt.colorbar()
    # plt.show()
end