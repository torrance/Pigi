mutable struct UVDatum{T <: AbstractFloat}
    row::Int
    chan::Int
    u::T
    v::T
    w::T
    weights::MMatrix{2, 2, T, 4}
    data::MMatrix{2, 2, Complex{T}, 4}
end
