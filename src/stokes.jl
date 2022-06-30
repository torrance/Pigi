stokesI(img::AbstractArray{SMatrix{2, 2, T, 4}}) where T = map(img) do x
    real(x[1, 1] + x[2, 2]) / 2
end