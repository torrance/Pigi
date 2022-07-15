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

@inline function Base.convert(::Type{LinearData{T}}, (Aleft, data, Aright)::Tuple{Comp2x2{T}, LinearData{T}, Comp2x2{T}}) where T
    # Calculate instrumental response
    # No normalization is performed, since we can't know which output we are using
    instr = (inv(Aleft) * data * inv(Aright)')

    return LinearData{T}(instr)
end

@inline function Base.convert(::Type{LinearData{T}}, data::StokesI{T}) where T
    return LinearData{T}(data.I, 0, 0, data.I)
end

@inline function Base.convert(::Type{LinearData{T}}, (Aleft, data, Aright)::Tuple{Comp2x2{T}, StokesI{T}, Comp2x2{T}}) where T
    # Normalize J based on Stokes I
    A2 = Aleft * Aright'
    norm = (real(A2[1, 1]) + real(A2[2, 2])) / 2

    # Calculate instrumental response
    I = data.I #  / norm
    instr = (Aleft * LinearData{T}(I, 0, 0, I) * Aright')

    return LinearData{T}(instr)
end

# TODO: Remove and set UVDatum.data::LinearData
@inline function Base.convert(::Type{Comp2x2{T}}, data::StokesI{T}) where T
    return Comp2x2{T}(data.I, 0, 0, data.I)
end

@inline function Base.convert(::Type{StokesI{T}}, data::LinearData{T}) where T
    return StokesI{T}((data.xx + data.yy) / 2)
end

@inline function Base.convert(::Type{StokesI{T}}, (Aleft, data, Aright)::Tuple{Comp2x2{T}, LinearData{T}, Comp2x2{T}}) where T
    # Normalize J based on Stokes I
    A2 = Aleft * Aright'
    norm = (real(A2[1, 1]) + real(A2[2, 2])) / 2

    # Calculate instrumental response
    instr = (inv(Aleft) * data * inv(Aright)') * norm

    return StokesI{T}((instr[1, 1] + instr[2, 2]) / 2)
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
