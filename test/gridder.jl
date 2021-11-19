@testset "Gridder" begin
    precision = Float64

    subgridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)
    padding = 14

    visgrid = zeros(Complex{precision}, 4, 128, 128)
    visgrid[:, 1 + padding:end - padding, 1 + padding:end - padding] = rand(Complex{precision}, 4, 128 - 2 * padding, 128 - 2 * padding)

    uvdata = Pigi.UVDatum{precision}[]
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

    taper = Pigi.mkkbtaper(subgridspec)

    subgrid = Pigi.Subgrid{precision}(
        0, 0, 0, 0, 0, subgridspec, Aleft, Aright, uvdata
    )

    idggrid = Pigi.gridder(subgrid)

    @test maximum(abs.(visgrid[1, :, :] .- idggrid[1, :, :])) < 1e-12
    @test maximum(abs.(visgrid[2, :, :] .- idggrid[2, :, :])) < 1e-12
    @test maximum(abs.(visgrid[3, :, :] .- idggrid[3, :, :])) < 1e-12
    @test maximum(abs.(visgrid[4, :, :] .- idggrid[4, :, :])) < 1e-12

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