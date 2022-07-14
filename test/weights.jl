@testset "Weighting ($(precision))" for precision in [Float32, Float64]
    uvdata = StructArray{Pigi.UVDatum{precision}}(undef, 0)

    gridspec = Pigi.GridSpec(128, 128, scaleuv=1.2)

    expectednatural = zeros(128, 128)

    for _ in 1:10_000
        upx, vpx = rand(2) * 100 .+ 14
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
    expectednatural ./= sum(expectednatural)

    weighter = Pigi.Natural(precision, uvdata, gridspec)
    weighteduvdata = deepcopy(uvdata)
    Pigi.applyweights!(weighteduvdata, weighter)

    naturalgrid = zeros(128, 128)
    for uvdatum in weighteduvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, naturalgrid, upx, vpx)
            naturalgrid[upx, vpx] += uvdatum.weights[1, 1]
        end
    end

    @test all(x -> isapprox(x[1], x[2], rtol=1e-5), zip(naturalgrid, expectednatural))

    weighter = Pigi.Uniform(precision, uvdata, gridspec)
    weighteduvdata = deepcopy(uvdata)
    Pigi.applyweights!(weighteduvdata, weighter)

    uniformgrid = zeros(128, 128)
    for uvdatum in weighteduvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, uniformgrid, upx, vpx)
            uniformgrid[upx, vpx] += uvdatum.weights[1, 1]
        end
    end

    # Unform weighting should be all ones (or empty cells)
    norm = sum(x -> x == 0 ? 0 : 1, uniformgrid)
    @test all(x -> x == 0 || isapprox(x, 1 / norm; rtol=1e-5), uniformgrid)

    weighter = Pigi.Briggs(precision, uvdata, gridspec, 2)
    weighteduvdata = deepcopy(uvdata)
    Pigi.applyweights!(weighteduvdata, weighter)

    briggsgrid = zeros(128, 128)
    for uvdatum in weighteduvdata
        upx, vpx = Pigi.lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if checkbounds(Bool, briggsgrid, upx, vpx)
            briggsgrid[upx, vpx] += uvdatum.weights[1, 1]
        end
    end

    # High briggs numbers should tend towards uniform weighting
    @test all(x -> isapprox(x[1], x[2], rtol=1e-2), zip(naturalgrid, briggsgrid))

    weighter = Pigi.Briggs(precision, uvdata, gridspec, 0)
    weighteduvdata = deepcopy(uvdata)
    Pigi.applyweights!(weighteduvdata, weighter)

    Aleft = Aright = ones(SMatrix{2, 2, Complex{precision}, 4}, 128, 128)
    subtaper = ones(precision, 128, 128)
    workunit = Pigi.WorkUnit{precision}(65, 65, 0, 0, 0, gridspec, Aleft, Aright, weighteduvdata)
    uvgrid = zeros(Pigi.LinearData{precision}, 128, 128)
    Pigi.gridder!(uvgrid, [workunit], subtaper; makepsf=true)

    xx = map(uv -> real(uv[1]), uvgrid)
    @test real.(fftshift(bfft(ifftshift(xx))))[65, 65] ≈ 1
end