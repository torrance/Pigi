@testset "Gridder on uniform grid (no w-terms)" begin
    precision = Float64

    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)
    padding = 14

    visgrid = zeros(Complex{precision}, 4, 128, 128)
    visgrid[:, 1 + padding:end - padding, 1 + padding:end - padding] = rand(Complex{precision}, 4, 128 - 2 * padding, 128 - 2 * padding)

    uvdata = StructArray{Pigi.UVDatum{precision}}(undef, 0)
    for vpx in axes(visgrid, 3), upx in axes(visgrid, 2)
        val = visgrid[:, upx, vpx]
        if !all(val .== 0)
            u, v = Pigi.px2lambda(upx, vpx, subgridspec)
            push!(uvdata, Pigi.UVDatum{precision}(
                0, 0, u, v, 0, [1 1; 1 1], val
            ))
        end
    end
    print("Gridding $(length(uvdata)) uvdatum")

    Aleft = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    subtaper = ones(precision, 128, 128)

    workunit = Pigi.WorkUnit{precision}(
        65, 65, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    idggrid = zeros(Pigi.LinearData{precision}, 128, 128)
    Pigi.gridder!(idggrid, [workunit], subtaper)
    idggrid = reinterpret(reshape, Complex{precision}, idggrid)

    @test maximum(abs.(visgrid[:, :, :] .- idggrid[:, :, :])) < 1e-12

    # plt.subplot(2, 2, 1)
    # plt.imshow(abs.(visgrid[1, :, :] .- idggrid[1, :, :]))
    # plt.colorbar()
    # plt.subplot(2, 2, 2)
    # plt.imshow(abs.(visgrid[2, :, :] .- idggrid[2, :, :]))
    # plt.colorbar()
    # plt.subplot(2, 2, 3)
    # plt.imshow(abs.(visgrid[3, :, :] .- idggrid[3, :, :]))
    # plt.colorbar()
    # plt.subplot(2, 2, 4)
    # plt.imshow(abs.(visgrid[4, :, :] .- idggrid[4, :, :]))
    # plt.colorbar()
    # plt.show()
end

@testset "Gridder on non-uniform grid (no w-terms)" begin
    precision = Float64
    N = 1000

    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)
    padding = 15

    center = (128 - 2 * padding) ÷ 2 + 1
    uvs = subgridspec.scaleuv * ((128 - 2 * padding) * rand(2, N) .- [center center;]')
    data = rand(SMatrix{2, 2, Complex{precision}, 4}, N)
    uvdata = StructArray{Pigi.UVDatum{precision}, 1}(undef, N)
    for i in eachindex(uvdata)
        u, v = uvs[:, i]
        uvdata[i] = Pigi.UVDatum(1, 1, u, v, zero(precision), @SMatrix(precision[1 1; 1 1]), data[i])
    end
    println("Gridding $(length(uvdata)) uvdatum")

    # We use a Gaussian taper, since we know its analytic Fourier representation
    sigmalm = 10 * subgridspec.scalelm
    taper = (l, m) -> exp(-(l^2 + m^2) / (2 * sigmalm^2))
    Taper = (rpx) -> 2π * sigmalm^2 * subgridspec.scaleuv^2 * exp(
        -2π^2 * sigmalm^2 * subgridspec.scaleuv^2 * rpx^2
    )

    # Creating expected gridded array, using direct convolution sampling
    expected = zeros(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    convolutionalsample!(expected, subgridspec, uvdata, Taper, padding)
    expected = reinterpret(reshape, Complex{precision}, expected)

    # Now run IDG gridding
    Aleft = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)

    subtaper = ones(precision, 128, 128)
    lms = fftfreq(subgridspec.Nx, 1 / subgridspec.scaleuv)
    for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
        subtaper[lpx, mpx] *= taper(l, m)
    end

    workunit = Pigi.WorkUnit{precision}(
        65, 65, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )
    idggrid = zeros(Pigi.LinearData{precision}, 128, 128)
    Pigi.gridder!(idggrid, [workunit], subtaper)
    idggrid = reinterpret(reshape, Complex{precision}, idggrid)

    @test maximum(abs.(expected .- idggrid)) < 1e-10

    # plt.subplot(1, 3, 1)
    # plt.imshow(abs.(expected[1, :, :]))
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.imshow(abs.(idggrid[1, :, :]))
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(abs.(expected[1, :, :] .- idggrid[1, :, :]))
    # plt.colorbar()
    # plt.show()
end

@testset "Gridder with non-unifrom grid (with w-terms)" begin
    precision = Float64

    subgridspec = Pigi.GridSpec(128, 128, 6u"arcminute")

    padding = 15
    N = 128

    # Source locations in radians, wrt to phase center, with x degree FOV
    sources = deg2rad.(
        rand(Float64, 2, 50) * 10 .- 5
    )
    sources[:, 1] .= 0

    println("Predicting UVDatum...")
    uvdata = StructArray{Pigi.UVDatum{precision}}(undef, 0)
    for (u, v, w) in eachcol(rand(Float64, 3, 1000))
        u = subgridspec.scaleuv * (u * (N - 2 * padding) .- ((N - 2 * padding) ÷ 2 + 1))
        v = subgridspec.scaleuv * (v * (N - 2 * padding) .- ((N - 2 * padding) ÷ 2 + 1))
        w = w * 25 - 12.6

        val = zero(SMatrix{2, 2, Complex{precision}, 4})
        for (ra, dec) in eachcol(sources)
            l, m = sin(ra), sin(dec)
            val += @SMatrix([1 0; 0 1]) * exp(-2π * 1im * (u * l + v * m + w * Pigi.ndash(l, m)))
        end
        push!(uvdata, Pigi.UVDatum{precision}(0, 0, u, v, w, @SMatrix(precision[1 1; 1 1]), val))
    end
    println("Done.")

    subtaper = Pigi.kaiserbessel(subgridspec, precision)

    # Calculate expected output
    expected = zeros(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)
    @time idft!(expected, uvdata, subgridspec, precision(length(uvdata)))
    expected .*= subtaper
    expected = reinterpret(reshape, Complex{precision}, expected)

    # Now run IDG gridding
    Aleft = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)
    Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, subgridspec.Nx, subgridspec.Ny)

    workunit = Pigi.WorkUnit{precision}(
        65, 65, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )
    idggrid = zeros(Pigi.LinearData{precision}, 128, 128)
    Pigi.gridder!(idggrid, [workunit], ifftshift(subtaper))
    idggrid = reinterpret(reshape, Complex{precision}, idggrid)
    idggrid = fftshift(ifft(fftshift(idggrid, (2, 3)), (2, 3)), (2, 3)) * subgridspec.Nx * subgridspec.Ny / length(uvdata)

    @test maximum(abs.(expected .- idggrid)) < 1e-14

    # plt.subplot(1, 3, 1)
    # plt.imshow(real.(expected[1, :, :]))
    # plt.colorbar()
    # plt.subplot(1, 3, 2)
    # plt.imshow(real.(idggrid[1, :, :]))
    # plt.colorbar()
    # plt.subplot(1, 3, 3)
    # plt.imshow(real.(expected[1, :, :] .- idggrid[1, :, :]))
    # plt.colorbar()
    # plt.show()
end