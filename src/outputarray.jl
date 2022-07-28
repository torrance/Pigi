abstract type OutputType{T, N, D} <: FieldArray{N, Complex{T}, D} end

struct LinearData{T} <: OutputType{T, Tuple{2, 2}, 2}
    xx::Complex{T}
    yx::Complex{T}
    xy::Complex{T}
    yy::Complex{T}
end

struct StokesI{T} <: OutputType{T, Tuple{}, 0}
    I::Complex{T}
end

function StokesI{T}(I::Tuple{Complex{T}}) where {T <: AbstractFloat}
    return StokesI{T}(I...)
end

#=
Conversion routines between output types and LinearData
=#

@inline function Base.convert(::Type{LinearData{T}}, data::StokesI{T}) where T
    return LinearData{T}(data.I, 0, 0, data.I)
end

# TODO: Remove and set UVDatum.data::LinearData
@inline function Base.convert(::Type{Comp2x2{T}}, data::StokesI{T}) where T
    return Comp2x2{T}(data.I, 0, 0, data.I)
end

@inline function Base.convert(::Type{StokesI{T}}, data::LinearData{T}) where T
    return StokesI{T}((data.xx + data.yy) / 2)
end

#=
Normalization used during inversion to mitigate errors and infinities from nulls
=#

function normalize(data::StokesI{T}, Aleft::Comp2x2{S}, Aright::Comp2x2{S})::StokesI{T} where {T, S}
    invAleft = inv(Aleft)
    invAright = inv(Aright')

    selectors = (
        Real2x2{T}(1, 0, 0, 0),
        Real2x2{T}(0, 1, 0, 0),
        Real2x2{T}(0, 0, 1, 0),
        Real2x2{T}(0, 0, 0, 1),
    )

    norm = zero(S)
    for selector in selectors
        J2 = invAleft * selector * invAright
        norm += abs(J2[1].re + J2[4].re) + abs(J2[1].im + J2[4].im)
    end

    return data / T(norm)
end

# Currently no normalization is performed for LinearData
function normalize(data::LinearData{T}, ::Comp2x2{T}, ::Comp2x2{T})::LinearData{T} where T
    return data
end

#=
FFT/IFFT wrappers for each OutputType
=#

function AbstractFFTs.fft!(arr::AbstractArray{LinearData{T}}) where T
    arr = reinterpret(reshape, Complex{T}, arr)
    return fft!(arr, (2, 3))
end

function AbstractFFTs.ifft!(arr::AbstractArray{LinearData{T}}) where T
    arr = reinterpret(reshape, Complex{T}, arr)
    return ifft!(arr, (2, 3))
end

function AbstractFFTs.fft!(arr::AbstractArray{StokesI{T}}) where T
    arr = reinterpret(Complex{T}, arr)
    return fft!(arr)
end

function AbstractFFTs.ifft!(arr::AbstractArray{StokesI{T}}) where T
    arr = reinterpret(Complex{T}, arr)
    return ifft!(arr)
end

#=
Other utility functions
=#

function Base.real(x::StokesI)
    return real(x.I)
end
