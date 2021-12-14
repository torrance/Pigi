@views function gaussian(xys, (xsigma, ysigma, pa))
    xs = @. xys[:, :, 1] * cos(pa) - xys[:, :, 2] * sin(pa)
    ys = @. xys[:, :, 1] * sin(pa) .+ xys[:, :, 2] * cos(pa)
    zs = @. exp.(-xs^2 / (2 * xsigma^2) - ys^2 / (2 * ysigma^2))
    return reshape(zs, :)
end

function psffit(psf)
    x0, y0 = size(psf) .÷ 2
    xys = map(CartesianIndices((-x0:x0 - 1, -y0:y0 - 1, 1:2))) do idx
        x, y, i = Tuple(idx)
        if i == 1
            return x
        else
            return y
        end
    end

    params = coef(curve_fit(gaussian, xys, reshape(psf, :), Float64[5, 5, 0]))

    fitted = reshape(gaussian(xys, params), size(psf))
    return params, fitted
end

function psfclip(psf, threshold)
    x0, y0 = size(psf) .÷ 2 .+ 1

    r02 = 0
    for idx in CartesianIndices(psf)
        x, y = Tuple(idx)
        r2 = (x - x0)^2 + (y - y0)^2

        if abs(psf[idx]) > threshold && r2 > r02
            r02 = r2
        end
    end

    r0 = ceil(Int, sqrt(r02) * 4)  # 4 is a magic number to ensure we get plenty of sidelobes
    r0 = r0 <= x0 - 1 ? r0 : x0 - 1
    return psf[x0 - r0:x0 + r0 - 1, y0 - r0:y0 + r0 - 1]
end