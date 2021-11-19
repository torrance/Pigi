struct UVDatum{T <: AbstractFloat}
    row::Int
    chan::Int
    u::T
    v::T
    w::T
    weights::SMatrix{2, 2, T, 4}
    data::SMatrix{2, 2, Complex{T}, 4}
end
