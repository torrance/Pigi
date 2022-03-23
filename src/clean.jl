function clean!(img, psf; gain=0.1, mgain=0.8, threshold=0., niter::Int=typemax(Int))
    components = similar(img)
    fill!(components, 0)

    imgd = CuArray(img)
    psfd = CuArray(psf)

    threshold = max((1 - mgain) * maximum(abs, img), threshold)
    println("Cleaning to threshold: $(threshold)")

    iter = 1
    while iter <= niter
        idx, val, absval = findabsmax(imgd)

        if absval < threshold
            println("\nThreshold limit reached ($(absval) < $(threshold))")
            break
        end

        if iter == 1 || mod(iter, 100) == 0
            print("\rClean iteration $(iter) found peak $(val) at $(idx)")
        end

        # Add the component to the component map
        xpeak, ypeak = Tuple(idx)
        components[xpeak, ypeak] += gain * val

        # Subtract out the psf
        subtractpsf(imgd, psfd, xpeak, ypeak, gain * val)

        iter += 1
    end

    copy!(img, imgd)
    CUDA.unsafe_free!(imgd)
    CUDA.unsafe_free!(psfd)

    return components, iter
end

function findabsmax(domain::AbstractArray{T}) where T
    function f(val, idx)
        return idx, val, abs(val)
    end

    function op(one, two)
        one[3] > two[3] && return one
        return two
    end

    idxs = CartesianIndices(domain)
    return mapreduce(
        f, op, domain, idxs;
        init=(idxs[1], zero(T), zero(typeof(abs(zero(T)))))
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

    @. img[xlow:xhigh, ylow:yhigh] -= f * psf[mlow:mhigh, nlow:nhigh]
end
