#pragma once

#include <hip/hip_runtime.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wuse-after-free"

/**
 * A shared pointer that only performs reference counting and deallocation on the host.
 * On the device, this simply acts as a pointer.
 *
 * This is currently used only for the metadata field on UVDatum.
 */
template <typename T>
class SharedHostPtr {
public:
    __host__ __device__
    SharedHostPtr() = default;

    __host__ __device__
    SharedHostPtr(T* ptr) : ptr(ptr) {
#ifndef __HIP_DEVICE_COMPILE__
        counter = new int {};
        ++(*counter);
#endif
    }

    __host__ __device__
    SharedHostPtr(const SharedHostPtr& other) {
        ptr = other.ptr;
        counter = other.counter;
#ifndef __HIP_DEVICE_COMPILE__
        if (counter) ++(*counter);
#endif
    }

    __host__ __device__
    SharedHostPtr(SharedHostPtr&& other) {
        *this = std::move(other);
    }

    __host__ __device__
    SharedHostPtr& operator=(const SharedHostPtr& other) {
#ifndef __HIP_DEVICE_COMPILE__
        if (counter && --(*counter) == 0) {
            delete ptr;
            delete counter;
        }
        if (other.counter) ++(*other.counter);
#endif

        ptr = other.ptr;
        counter = other.counter;
        return *this;
    }

    __host__ __device__
    SharedHostPtr& operator=(SharedHostPtr&& other) {
        using std::swap;
        swap(ptr, other.ptr);
        swap(counter, other.counter);
        return *this;
    }

    __host__ __device__
    ~SharedHostPtr() {
#ifndef __HIP_DEVICE_COMPILE__
        if (counter && --(*counter) == 0) {
            delete ptr;
            delete counter;
        }
#endif
    }

    T* operator->() { return ptr; }

private:
    T* ptr {};
    int* counter {};
};

template<typename T, typename... Args>
SharedHostPtr<T> makesharedhost(Args... args) {
    T* ptr = new T(std::forward<Args>(args)...);
    return SharedHostPtr<T>(ptr);
}

#pragma GCC diagnostic pop