@testset "Weighting" begin
    precision = Float64
    uvdata = Pigi.UVDatum{precision}[]

    gridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)

    expectednatural = zeros(128, 128)

    for _ in 1:1000
        upx, vpx = rand(2) * 128
        u, v = Pigi.px2lambda(upx, vpx, gridspec)
        push!(uvdata, Pigi.UVDatum{precision}(
            0, 0, u, v, 0, rand(SMatrix{2, 2, Float64, 4}), [1 1; 1 1]
        ))

        # Construct expected natural weight grid
        upx, vpx = round(Int, upx), round(Int, vpx)
        if checkbounds(Bool, expectednatural, upx, vpx)
            expectednatural[upx, vpx] += uvdata[end].weights[1, 1]
        end
    end

    weighter = Pigi.Natural(uvdata, gridspec)
    naturalgrid = zeros(128, 128)
    for uvdatum in uvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, naturalgrid, upx, vpx)
            weight = uvdatum.weights .* weighter(uvdatum)
            naturalgrid[upx, vpx] += weight[1, 1]
        end
    end

    @test all(x -> isapprox(x[1], x[2]), zip(naturalgrid, expectednatural))

    weighter = Pigi.Uniform(uvdata, gridspec)
    uniformgrid = zeros(128, 128)
    for uvdatum in uvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, uniformgrid, upx, vpx)
            weight = uvdatum.weights .* weighter(uvdatum)
            uniformgrid[upx, vpx] += weight[1, 1]
        end
    end

    # Unform weighting should be all ones (or empty cells)
    @test all(x -> x == 0 || x ≈ 1, uniformgrid)

    weighter = Pigi.Briggs(uvdata, gridspec, 4)
    briggsgrid = zeros(128, 128)
    for uvdatum in uvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, briggsgrid, upx, vpx)
            weight = uvdatum.weights .* weighter(uvdatum)
            briggsgrid[upx, vpx] += weight[1, 1]
        end
    end

    # High briggs numbers should tend towards uniform weighting
    @test all(x -> isapprox(x[1], x[2], atol=1e-5), zip(naturalgrid, briggsgrid))
end