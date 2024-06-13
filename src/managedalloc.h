#pragma once

#include <hip/hip_runtime.h>

#include "hip.h"

template <typename T>
class ManagedAllocator {
public:
    using value_type = T;

    ManagedAllocator() noexcept = default;

    template <typename U>
    ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

    bool operator==(ManagedAllocator<T>&) { return true; }
    bool operator!=(ManagedAllocator<T>&) { return false; }

    template <typename U>
    struct rebind { using other = ManagedAllocator<U>; };

    T* allocate(size_t n) {
        T* ptr;
        HIPCHECK( hipMallocManaged(&ptr, n * sizeof(T)) );
        return ptr;
    }

    void deallocate(T* ptr, size_t) {
        HIPCHECK( hipFree(ptr) );
    }
};