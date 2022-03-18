@testset "In place fftshift" begin
    arr = rand(ComplexF64, 10, 10)
    expected = fftshift(arr)
    Pigi.fftshift!(arr)
    @test all(expected .== arr)

    arr = rand(SMatrix{2, 2, ComplexF64, 4}, 4000, 4000)
    arrd = CuArray(arr)
    expected = fftshift(arr)
    Pigi.fftshift!(arr)
    Pigi.fftshift!(arrd)
    @test all(x == y for (x, y) in zip(arr, expected))
    @test all(x == y for (x, y) in zip(Array(arrd), expected))
end