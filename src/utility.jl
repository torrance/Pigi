"""
    Calculate n' = 1 - n
                 = 1 - √(1 - l^2 - m^2)
                 = (l^2 + m^2) / (1 + √(1 - l^2 - m^2))

    The final line is (apparently) more accurate for small values of n.

    Note for values of n > 1, which are unphysical, ndash is set to 1.
"""
@inline function ndash(l::T, m::T) where {T}
    r2 = l^2 + m^2
    if r2 > 1
        return T(1)
    else
        return r2 / (1 + sqrt(1 - r2))
    end
end

function fftshift!(arr::AbstractMatrix)
    @assert all(iseven(x) for x in size(arr)) "fftshift!() must operate on arrays with even dimensions"

    Nx, Ny = size(arr)
    shiftx, shifty = Nx ÷ 2, Ny ÷ 2

    for j1 in 1:shifty
        for i1 in 1:Nx
            i2 = mod1(i1 + shiftx, Nx)
            j2 = j1 + shifty
            arr[i1, j1], arr[i2, j2] = arr[i2, j2], arr[i1, j1]
        end
    end
end

function fftshift!(arr::CuMatrix)
    @assert all(iseven(x) for x in size(arr)) "fftshift!() must operate on arrays with even dimensions"

    function _fftshift!(arr::CuDeviceMatrix, shiftx, shifty, Nx)
        idx = blockDim().x * (blockIdx().x - 1) + threadIdx().x
        if idx > Nx * shifty
            return nothing
        end

        i1 = mod1(idx, Nx)
        j1 = div(idx - 1, Nx) + 1

        i2 = mod1(i1 + shiftx, Nx)
        j2 = j1 + shifty

        arr[i1, j1], arr[i2, j2] = arr[i2, j2], arr[i1, j1]

        return nothing
    end

    Nx, Ny = size(arr)
    shiftx, shifty = Nx ÷ 2, Ny ÷ 2

    kernel = @cuda launch=false _fftshift!(arr, shiftx, shifty, Nx)
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, Nx * shifty)
    blocks = cld(Nx * shifty, threads)
    kernel(arr, shiftx, shifty, Nx; threads, blocks)
end

function permute2vector(input::NTuple{N, Array{T, D}}) where {N, T, D}
    output = Array{SVector{N, T}, D}(undef, size(input[1]))

    for (i, vals) in enumerate(zip(input...))
        output[i] = vals
    end

    return output
end

function permute2vector(input)
    return permute2vector(tuple(input...))
end