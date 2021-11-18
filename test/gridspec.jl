@testset "Coordinate conversions" begin
    for (ulambda, vlambda) in eachcol(100 .* rand(2, 100))
        gridspec = Pigi.GridSpec(3000, 3000, scaleuv=rand() + 0.5)
        upx, vpx = Pigi.lambda2px(ulambda, vlambda, gridspec)
        ulambdaagain, vlambdaagain = Pigi.px2lambda(upx, vpx, gridspec)
        @test ulambda ≈ ulambdaagain
        @test vlambda ≈ vlambdaagain
    end
end