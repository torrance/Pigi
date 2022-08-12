@testset "Simple clean" for precision in [Float32, Float64]
    expectedcomponentmap = zeros(SVector{1, precision}, 1000, 1000)

    for (xpx, ypx) in eachcol(rand(1:1000, 2, 1000))
        expectedcomponentmap[xpx, ypx] += SVector{1, precision}(rand())
    end

    psf = map(CartesianIndices((-35:34, -35:34))) do idx
        sigmax = 5
        sigmay = 10
        x, y = Tuple(idx)
        return SVector{1, precision}(exp(-x^2 / (2 * sigmax^2) - y^2 / (2 * sigmay^2)))
    end

    expectedcomponentmap_flat = reinterpret(precision, expectedcomponentmap)
    psf_flat = reinterpret(precision, psf)
    img = similar(expectedcomponentmap)
    img_flat = reinterpret(precision, img)
    img_flat .= conv(expectedcomponentmap_flat, psf_flat)[1 + 35:end - 34, 1 + 35:end - 34]

    componentmap, iter = Pigi.clean!(copy(img), psf, [100e6], GPUArray; mgain=1, threshold=1e-2)


    restored = conv(reinterpret(precision, componentmap), psf_flat)[1 + 35:end - 34, 1 + 35:end - 34]
    diff = img_flat .- restored

    @test all(x -> abs(x) < 1.2e-2, diff)

    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(img))
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(restored))
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(real.(diff))
    # plt.colorbar()
    # plt.show()
end

@testset "MFS Clean" for precision in [Float32, Float64]
    freqs = Float64[0, 0.05, 0.12, 0.17]

    expectedcomponentmap = zeros(SVector{4, precision}, 1000, 1000)

    for (xpx, ypx) in eachcol(rand(1:1000, 2, 50))
        vals = rand() .+ 1 * rand() * freqs  # Use a linear spectral index
        expectedcomponentmap[xpx, ypx] += SVector{4, precision}(vals)
    end

    psf = map(CartesianIndices((-35:34, -35:34))) do idx
        sigmax = [5, 4.5, 4.2, 4]
        sigmay = [10, 9, 8.4, 8]
        x, y = Tuple(idx)
        return SVector{4, precision}(exp.(-x^2 ./ (2 * sigmax.^2) - y^2 ./ (2 * sigmay.^2)))
    end
    psf_flat = reinterpret(reshape, precision, psf)

    expectedcomponentmap_flat = reinterpret(reshape, precision, expectedcomponentmap)

    img = similar(expectedcomponentmap)
    img_flat = reinterpret(reshape, precision, img)
    for i in 1:4
        img_flat[i, :, :] .= conv(expectedcomponentmap_flat[i, :, :], psf_flat[i, :, :])[1 + 35:end - 34, 1 + 35:end - 34]
    end

    componentmap, iter = Pigi.clean!(copy(img), psf, freqs, GPUArray; mgain=1, threshold=5e-3, gain=0.01, degree=1)
    componentmap_flat = reinterpret(reshape, precision, componentmap)

    restored = similar(img_flat)
    for i in 1:4
        restored[i, :, :] .= conv(componentmap_flat[i, :, :], psf_flat[i, :, :])[1 + 35:end - 34, 1 + 35:end - 34]
    end

    diff = img_flat .- restored

    println(maximum(x -> abs(x), diff))
    @test maximum(x -> abs(x), diff) < 1e-2
end

@testset "findabsmax()" for precision in [Float32, Float64]
    arr = rand(SVector{4, precision}, 9000, 9000)
    arrd = GPUArray(arr)

    val1, idx1 = findmax(abs ∘ sum, arr)
    idx2, val2, absval2 = Pigi.findabsmax(arrd)

    @test idx1 == idx2
    @test val1 == absval2
end