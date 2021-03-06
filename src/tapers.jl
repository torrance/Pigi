function kaiserbessel(gridspec::GridSpec, ::Type{Float32}; alpha=4.2)::Matrix{Float32}
    return kaiserbessel(gridspec, Float32, alpha)
end

function kaiserbessel(gridspec::GridSpec, ::Type{Float64}; alpha=10)::Matrix{Float64}
    return kaiserbessel(gridspec, Float64, alpha)
end

function kaiserbessel(gridspec::GridSpec, ::Type{T}, alpha)::Matrix{T} where {T <: AbstractFloat}
    function kb1D(n, N)
        x = n / N - 0.5

        if x < -0.5 || x > 0.5
            return 0.
        else
            return besseli(0, π * alpha * sqrt(1 - 4 * x^2)) / besseli(0, π * alpha)
        end
    end

    return map(CartesianIndices((1:gridspec.Nx,  1:gridspec.Ny))) do xy
        x, y = Tuple(xy)
        return kb1D(x - 1.5, gridspec.Nx - 1) * kb1D(y - 1.5, gridspec.Ny - 1)
    end
end

function blackmanharris(gridspec::GridSpec, ::Type{T})::Matrix{T} where {T <: AbstractFloat}
    function blackmanharris1D(n, N)
        if n < 0 || n >= N
            return 0.
        else
            return 0.35875 - 0.48829 * cos(2π * (n / N)) + 0.14128 * cos(4π * (n / N)) - 0.01168 * cos(6π * (n / N))
        end
    end

    return map(CartesianIndices((1:gridspec.Nx, 1:gridspec.Ny))) do xy
        x, y = Tuple(xy)
        return blackmanharris1D(x - 1.5, gridspec.Nx - 1) * blackmanharris1D(y - 1.5, gridspec.Ny - 1)
    end
end