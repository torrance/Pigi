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

@testset "permute2vector()" begin
    input = (rand(7000, 7000), rand(7000, 7000), rand(7000, 7000), rand(7000, 7000))

    output = Pigi.permute2vector(input)  # Tuple of matrices
    output_flat = reinterpret(reshape, Float64, output)

    @test typeof(output) == Array{SVector{4, Float64}, 2}
    @test size(output) == (7000, 7000)
    @test output_flat[1, :, :] == input[1]
    @test output_flat[2, :, :] == input[2]
    @test output_flat[3, :, :] == input[3]
    @test output_flat[4, :, :] == input[4]

    @test output == Pigi.permute2vector(collect(input))  # Array of matrices
end