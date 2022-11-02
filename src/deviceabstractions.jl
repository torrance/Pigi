function kernelconf(::Union{Array, SubArray})
    return CPU(), KernelAbstractions.NDIteration.DynamicSize(), KernelAbstractions.NDIteration.DynamicSize() # , min(Threads.nthreads(), length(a))
end

function kernelconf(::CuArray)
    return CUDADevice(), KernelAbstractions.NDIteration.DynamicSize(), KernelAbstractions.NDIteration.DynamicSize()
end

function kernelconf(a::ROCArray)
    return ROCDevice(), min(1024, length(a))
end

function pagelock(::Type{Array}, a::Array)
    return a
end

function pagelock(::Type{CuArray}, a::Array)
    return CUDA.Mem.pin(a)
end

function pagelock(::Type{ROCArray}, a::Array)
    ptr = AMDGPU.Mem.lock(a)
    finalizer(a) do _
        AMDGPU.Mem.unlock(ptr)
    end

    return a
end

function getwrapper(::Union{Array, SubArray})
    return Array
end

function getwrapper(::CuArray)
    return CuArray
end

function getwrapper(::ROCArray)
    return ROCArray
end
