@testset "In place fftshift" begin
    arr = rand(ComplexF64, 10, 10)
    expected = fftshift(arr)
    Pigi.fftshift!(arr)
    @test all(expected .== arr)

    arr = rand(SMatrix{2, 2, ComplexF64, 4}, 4000, 4000)
    expected = fftshift(arr)
    Pigi.fftshift!(arr)
    @test all(x == y for (x, y) in zip(arr, expected))
end