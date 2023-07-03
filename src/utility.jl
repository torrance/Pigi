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

    @kernel function _fftshift!(arr::AbstractMatrix, shiftx, shifty, Nx)
        idx = @index(Global)

        i1 = mod1(idx, Nx)
        j1 = div(idx - 1, Nx) + 1

        i2 = mod1(i1 + shiftx, Nx)
        j2 = j1 + shifty

        arr[i1, j1], arr[i2, j2] = arr[i2, j2], arr[i1, j1]
    end

    Nx, Ny = size(arr)
    shiftx, shifty = Nx ÷ 2, Ny ÷ 2

    kernel = _fftshift!(kernelconf(arr)...)
    kernel(arr, shiftx, shifty, Nx; ndrange=Nx * shifty)
    KernelAbstractions.synchronize(kernelconf(arr)[begin])
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

function resample(img::AbstractArray{Complex{T}}, fromgrid::GridSpec, togrid::GridSpec) where {T <: Real}
    @assert fromgrid.scaleuv == togrid.scaleuv

    Img = fftshift(ifft(img))
    ImgResampled = zeros(Complex{T}, togrid.Nx, togrid.Ny)

    u0, v0 = togrid.Nx ÷ 2 + 1, togrid.Ny ÷ 2 + 1
    uwidth, vwidth = fromgrid.Nx ÷ 2, fromgrid.Ny ÷ 2

    ImgResampled[u0 - uwidth:u0 + uwidth - 1, v0 - vwidth:v0 + vwidth - 1] .= Img
    return fft(ifftshift(ImgResampled))
end

function resample(img::AbstractArray{T}, fromgrid::GridSpec, togrid::GridSpec) where {T <: Real}
    img = complex(img)
    return real.(resample(img::AbstractArray{Complex{T}}, fromgrid::GridSpec, togrid::GridSpec))
end

function resample(img::AbstractArray{SMatrix{2, 2, T, 4}}, fromgrid::GridSpec, togrid::GridSpec) where {T <: Complex}
    resampled = Array{SMatrix{2, 2, T, 4}, 2}(undef, togrid.Nx, togrid.Ny)

    img_flattened = reinterpret(reshape, T, img)
    resampled_flattened = reinterpret(reshape, T, resampled)

    for i in 1:4
        resampled_flattened[i, :, :] = resample(img_flattened[i, :, :], fromgrid, togrid)
    end

    return resampled
end
