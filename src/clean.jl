function clean!(img::Array{SVector{N, T}, 2}, psf::Array{SVector{N, T}, 2}, freqs::Vector, ::Type{wrapper}; gain=0.1, mgain=0.8, threshold=0., niter::Int=typemax(Int), degree=-1) where {N, T, wrapper}
    components = similar(img)
    fill!(components, zero(SVector{N, T}))

    imgd = wrapper(img)
    psfd = wrapper(psf)

    threshold = max((1 - mgain) * maximum(abs ∘ sum, img) / N, threshold)
    println("Cleaning to threshold: $(threshold)")

    iter = 1
    while iter <= niter
        idx, val, absval = findabsmax(imgd)
        absval /= N

        if absval < threshold
            println("\nThreshold limit reached ($(absval) < $(threshold))")
            break
        end

        if iter == 1 || mod(iter, 100) == 0
            print("\rClean iteration $(iter) found peak $(sum(val) / N) at $(idx)")
        end

        # If degree == 0, it's just the mean of each channel
        if degree == 0
            val = gain * SVector{N, T}(mean(s), mean(s), mean(s), mean(s))
        # If the degree is greater than or equal to the number of data points, then we
        # are overfitting. So just return the channel values. degree < 0 is treated the same.
        elseif degree >= N || degree < 0
            val = gain * val
        else
            p = fit(freqs, val, degree)
            val = SVector{N, T}(gain * p.(freqs))
        end

        # Add the component to the component map
        xpeak, ypeak = Tuple(idx)
        components[xpeak, ypeak] += val

        # Subtract out the psf
        subtractpsf(imgd, psfd, xpeak, ypeak, val)

        iter += 1
    end

    copy!(img, imgd)

    return components, iter
end

function findabsmax(domain::AbstractArray{SVector{N, T}, 2}) where {N, T}
    function f(val, idx)
        return idx, val, (abs ∘ sum)(val)
    end

    function op(one, two)
        one[3] > two[3] && return one
        return two
    end

    idxs = CartesianIndices(domain)
    return mapreduce(
        f, op, domain, idxs;
        init=(idxs[1], zero(SVector{N, T}), zero(typeof(abs(zero(T)))))
    )
end

function subtractpsf(img, psf, xpeak, ypeak, f)
    # x, y are coordinates into img
    # m, n are coordinates into psf

    m0, n0 = size(psf) .÷ 2 .+ 1

    mlow, mhigh = 1, size(psf, 1)
    nlow, nhigh = 1, size(psf, 2)

    xlow = xpeak - m0 + 1
    xhigh = xpeak + mhigh - m0
    ylow = ypeak - n0 + 1
    yhigh = ypeak + nhigh - n0

    # Now handle all the edge cases where the PSF map
    # escapes the borders of the img.
    if xlow < 1
        mlow += 1 - xlow
        xlow = 1
    end

    if ylow < 1
        nlow += 1 - ylow
        ylow = 1
    end

    if xhigh > size(img, 1)
        mhigh -= xhigh - size(img, 1)
        xhigh = size(img, 1)
    end

    if yhigh > size(img, 2)
        nhigh -= yhigh - size(img, 2)
        yhigh = size(img, 2)
    end

    @. img[xlow:xhigh, ylow:yhigh] -= broadcast(.*, (f,), psf[mlow:mhigh, nlow:nhigh])
end
