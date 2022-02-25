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
            println("\nThreshold limit reached ($(absval) < $(threshold))")
            break
        end

        if iter == 1 || mod(iter, 100) == 0
            print("\rClean iteration $(iter) found peak $(val) at $(idx)")
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
    buffer = CuVector{@NamedTuple{idx::Int, val::T, absval::P}}(undef, 0)
    resultd = CuVector{@NamedTuple{idx::Int, val::T, absval::P}}(undef, 1)

    kernel = @cuda launch=false _findabsmax(domain, buffer)
    config = launch_configuration(kernel.fun)
    threads = min(length(domain), config.threads)
    blocks = cld(length(domain), 2 * threads)
    buffer = CuVector{@NamedTuple{idx::Int, val::T, absval::P}}(undef, blocks)
    CUDA.@sync kernel(
        domain, buffer; threads, blocks,
        shmem=sizeof(@NamedTuple{idx::Int, val::T, absval::P}) * threads
    )

    kernel = @cuda launch=false _findabsmaxblockreduction(buffer, resultd)
    config = launch_configuration(kernel.fun)
    threads = min(length(buffer), config.threads)
    CUDA.@sync kernel(
        buffer, resultd; threads, blocks=1,
        shmem=sizeof(@NamedTuple{idx::Int, val::T, absval::P}) * threads
    )

    result = only(Array(resultd))
    CUDA.unsafe_free!(resultd)
    CUDA.unsafe_free!(buffer)
    return result
end

function _findabsmax(
    domain::CuDeviceArray{T},
    buffer::CuDeviceVector{@NamedTuple{idx::Int, val::T, absval::P}}
) where {T, P}
    shm = CUDA.CuDynamicSharedArray(@NamedTuple{idx::Int, val::T, absval::P}, blockDim().x)

    idx1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx2 = gridDim().x * blockDim().x + (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx1 > length(domain)
        shm[threadIdx().x] = (idx=-1, val=-1, absval=-1)
    elseif idx2 > length(domain)
        val1 = domain[idx1]
        shm[threadIdx().x] = (idx=idx1, val=val1, absval=abs(val1))
    else
        val1 = domain[idx1]
        val2 = domain[idx2]
        absval1 = abs(val1)
        absval2 = abs(val2)

        if absval1 > absval2
            shm[threadIdx().x] = (idx=idx1, val=val1, absval=absval1)
        else
            shm[threadIdx().x] = (idx=idx2, val=val2, absval=absval2)
        end
    end

    CUDA.sync_threads()

    s, n = divrem(blockDim().x, 2)
    while s > 0
        if threadIdx().x <= s + n
            idxval1, idxval2 = shm[threadIdx().x], shm[threadIdx().x + s]
            if idxval1.absval > idxval2.absval
                shm[threadIdx().x] = idxval1
            else
                shm[threadIdx().x] = idxval2
            end
        end
        s, n = divrem(s + n, 2)
        CUDA.sync_threads()
    end

    buffer[blockIdx().x] = shm[1]

    return nothing
end

function _findabsmaxblockreduction(
    buffer::CuDeviceVector{@NamedTuple{idx::Int, val::T, absval::P}},
    result::CuDeviceVector{@NamedTuple{idx::Int, val::T, absval::P}}
) where {T, P}
    shm = CUDA.CuDynamicSharedArray(@NamedTuple{idx::Int, val::T, absval::P}, blockDim().x)

    maxval = buffer[threadIdx().x]
    for idx in threadIdx().x + blockDim().x:blockDim().x:length(buffer)
        val = buffer[idx]
        if val.absval > maxval.absval
            maxval = val
        end
    end

    shm[threadIdx().x] = maxval

    CUDA.sync_threads()

    s, n = divrem(blockDim().x, 2)
    while s > 0
        if threadIdx().x <= s + n
            idxval1, idxval2 = shm[threadIdx().x], shm[threadIdx().x + s]
            if idxval1.absval > idxval2.absval
                shm[threadIdx().x] = idxval1
            else
                shm[threadIdx().x] = idxval2
            end
        end
        s, n = divrem(s + n, 2)
        CUDA.sync_threads()
    end

    if threadIdx().x == 1
        result[1] = shm[1]
    end

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
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > length(psf)
        return nothing
    end

    mpx, npx = Tuple(CartesianIndices(psf)[idx])
    xpx = xpeak + mpx - m0
    ypx = ypeak + npx - n0

    if 1 <= xpx <= size(img, 1) && 1 <= ypx <= size(img, 2)
        img[xpx, ypx] -= f * psf[mpx, npx]
    end

    return nothing
end
