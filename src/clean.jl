function clean!(img, psf; gain=0.1, mgain=0.8, threshold=0, niter::Int=typemax(Int))
    components = similar(img)
    fill!(components, 0)

    imgd = CuArray(img)
    psfd = CuArray(psf)

    threshold = maximum([(1 - mgain) * maximum(abs, img), threshold])
    println("Cleaning to threshold: $(threshold)")

    timemax = zero(UInt64)
    timesubtract = zero(UInt64)

    iter = 1
    while iter <= niter
        start = time_ns()
        idx, val, absval = findabsmax(imgd)
        timemax += time_ns() - start

        if absval < threshold
            println("Threshold limit reached ($(absval) < $(threshold))")
            break
        end

        if iter == 1 || mod(iter, 100) == 0
            println("Clean iteration $(iter) found peak $(val) at $(idx)")
        end

       # Add the component to the component map
        xpeak, ypeak = Tuple(CartesianIndices(img)[idx])
        components[xpeak, ypeak] += gain * val

         # Subtract out the psf
        start = time_ns()
        subtractpsf(imgd, psfd, xpeak, ypeak, gain * val)
        timesubtract += time_ns() - start

        iter += 1
    end

    println("Cleaning time budget: $(timemax / 1e9) s peak searching, $(timesubtract / 1e9) s PSF subtracting")

    copy!(img, imgd)
    return components, iter
end

function findabsmax(domain::CuArray{T}) where T
    P = typeof(abs(zero(T)))
    result = CuVector{@NamedTuple{idx::Int, val::T, absval::P}}(undef, 1)

    kernel = @cuda launch=false _findabsmax(result, domain)
    config = launch_configuration(kernel.fun)
    nthreads = min(length(domain), config.threads)
    kernel(
        result, domain;
        threads=nthreads,
        blocks=1,
        shmem=sizeof(@NamedTuple{idx::Int, val::T, absval::P}) * nthreads
    )
    return only(Array(result))
end

function _findabsmax(result::CuDeviceVector{@NamedTuple{idx::Int, val::T, absval::P}}, domain::CuDeviceArray{T}) where {T, P}
    shm = CUDA.CuDynamicSharedArray(@NamedTuple{idx::Int, val::T, absval::P}, blockDim().x)

    idx = threadIdx().x
    stride = blockDim().x

    val = domain[idx]
    absval = abs(val)
    maxidxval = (idx=idx, val=val, absval=absval)

    for idx in idx + stride:stride:length(domain)
        val = domain[idx]
        absval = abs(val)
        if absval > maxidxval.absval
            maxidxval = (idx=idx, val=val, absval=absval)
        end
    end
    shm[idx] = maxidxval

    CUDA.sync_threads()

    s = 1
    while s < blockDim().x
        i = 2 * (threadIdx().x - 1) * s + 1
        if i + s <= blockDim().x
            idxval1, idxval2 = shm[i], shm[i + s]
            if idxval1.absval > idxval2.absval
                shm[i] = idxval1
            else
                shm[i] = idxval2
            end
        end
        s *= 2
        CUDA.sync_threads()
    end

    result[1] = shm[1]
    return nothing
end

function subtractpsf(img, psf, xpeak, ypeak, f)
    n0, m0 = size(psf) .÷ 2 .+ 1

    kernel = @cuda launch=false _subtractpsf(img, psf, xpeak, ypeak, f, n0, m0)
    config = launch_configuration(kernel.fun)
    threads = min(length(psf), config.threads)
    blocks = cld(length(psf), threads)
    kernel(img, psf, xpeak, ypeak, f, n0, m0; threads, blocks)
end

function _subtractpsf(img, psf, xpeak, ypeak, f, n0, m0)
    mpx, npx = Tuple(CartesianIndices(psf)[(blockIdx().x - 1) * blockDim().x + threadIdx().x])
    xpx = xpeak + mpx - m0
    ypx = ypeak + npx - n0

    if 1 <= xpx <= size(img, 1) && 1 <= ypx <= size(img, 2)
        img[xpx, ypx] -= f * psf[mpx, npx]
    end

    return nothing
end
