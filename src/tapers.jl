function mkkbtaper(gridspec; alpha=14)
    # We want the KB function to reach zero at the edge of our image. rnorm ensures this.
    rnorm = (minimum([gridspec.Nx, gridspec.Ny]) ÷ 2 - 1) * gridspec.scalelm

    kbnorm = besseli(0, π * alpha)
    function taper(l, m)
        l /= 2 * rnorm
        m /= 2 * rnorm
        r2 = l^2 + m^2
        if r2 > 0.25
            return 0.
        else
            return besseli(0, π * alpha * sqrt(1 - 4 * r2)) / kbnorm
        end
    end
    return taper
end

function applytaper!(img, gridspec, taper)
    Threads.@threads for lm in CartesianIndices(img)
        lpx, mpx = Tuple(lm)
        l, m = px2sky(lpx, mpx, gridspec)
        img[lm] *= taper(l, m)
    end
end

function removetaper!(img, gridspec, taper)
    Threads.@threads for lm in CartesianIndices(img)
        lpx, mpx = Tuple(lm)
        l, m = px2sky(lpx, mpx, gridspec)
        img[lm] /= taper(l, m)
    end
end