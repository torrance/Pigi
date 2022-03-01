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

function fftshift!(arr::AbstractArray{T, 2}) where {T}
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