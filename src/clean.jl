function clean!(img, psf; gain=0.1, mgain=0.8, threshold=0, niter::Int=typemax(Int))
    components = similar(img)
    fill!(components, 0)

    absimg = mappedarray(abs, img)

    threshold = maximum([(1 - mgain) * maximum(absimg), threshold])
    println("Cleaning to threshold: $(threshold)")

    n0, m0 = size(psf) .÷ 2 .+ 1

    timemax = zero(UInt64)
    timesubtract = zero(UInt64)

    iter = 1
    while iter <= niter
        start = time_ns()
        absval, idx = findmax(absimg)
        timemax += time_ns() - start

        if absval < threshold
            println("Threshold limit reached ($(absval) < $(threshold))")
            break
        end
        val = img[idx]

        if iter == 1 || mod(iter, 100) == 0
            println("Clean iteration $(iter) found peak $(val) at $(idx)")
        end

        # Subtract out the psf
        xpeak, ypeak = Tuple(idx)
        components[xpeak, ypeak] += gain * val

        start = time_ns()
        for npx in axes(psf, 2)
            ypx = ypeak + npx - n0
            if 1 <= ypx <= size(img, 2)
                for mpx in axes(psf, 1)
                    xpx = xpeak + mpx - m0
                    if 1 <= xpx <= size(img, 1)
                        img[xpx, ypx] -= gain * val * psf[npx, mpx]
                    end
                end
            end
        end
        timesubtract += time_ns() - start

        iter += 1
    end

    println("Cleaning time budget: $(timemax / 1e9) s peak searching, $(timesubtract / 1e9) s PSF subtracting")

    return components, iter
end