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