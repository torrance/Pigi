@testset "Simple clean" begin
    expectedcomponentmap = zeros(ComplexF32, 1000, 1000)

    for (xpx, ypx) in eachcol(rand(1:1000, 2, 1000))
        expectedcomponentmap[xpx, ypx] += rand()
    end

    psf = map(CartesianIndices((-35:34, -35:34))) do idx
        sigmax = 5
        sigmay = 10
        x, y = Tuple(idx)
        return Float32(exp(-x^2 / (2 * sigmax^2) - y^2 / (2 * sigmay^2)))
    end

    img = conv(expectedcomponentmap, psf)[1 + 35:end - 34, 1 + 35:end - 34]

    componentmap, iter = Pigi.clean!(copy(img), psf, mgain=1, threshold=1e-2)

    restored = conv(componentmap, psf)[1 + 35:end - 34, 1 + 35:end - 34]
    diff = img .- restored

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

@testset "findabsmax()" begin
    arr = rand(9000, 9000)
    arrd = CuArray(arr)

    val1, idx1 = findmax(abs, arr)
    idx2, val2, absval2 = Pigi.findabsmax(arrd)

    @test idx1 == idx2
    @test val1 == absval2
end