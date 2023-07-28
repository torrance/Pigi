#pragma once

#include <array>
#include <algorithm>
#include <vector>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

#include "hip.cpp"

enum class MemoryStorage { Host, Device };

template <typename T, MemoryStorage M>
class BasePointer {
public:
    __host__ __device__ BasePointer() = default;
    __host__ __device__ BasePointer(T* ptr) : ptr(ptr) {}

    __host__ __device__ inline T& operator*() const { return *ptr; }
    __host__ __device__ inline T** operator&() { return &ptr; }

    __host__ __device__ inline T& operator[](size_t i) const { return ptr[i]; }

    __host__ __device__ inline auto operator+(auto x) const { return BasePointer {ptr + x}; }

    __host__ __device__ inline operator T*() const { return ptr; }
    __host__ __device__ inline operator bool() const { return (bool) ptr; }

    friend void swap(BasePointer<T, M>& lhs, BasePointer<T, M>& rhs) noexcept {
        using std::swap;
        swap(lhs.ptr, rhs.ptr);
    }

private:
    T* ptr;
};

template <typename T>
using HostPointer = BasePointer<T, MemoryStorage::Host>;

template <typename T>
using DevicePointer = BasePointer<T, MemoryStorage::Device>;

template <typename T>
void malloc(HostPointer<T>& ptr, size_t sz) {
    HIPCHECK( hipHostMalloc(&ptr, sz) );
}

template <typename T>
void malloc(DevicePointer<T>& ptr, size_t sz) {
    HIPCHECK( hipMallocAsync(&ptr, sz, hipStreamPerThread) ); }

template <typename T>
void memcpy(HostPointer<T> dst, HostPointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyHostToHost, hipStreamPerThread)
    );
}

template <typename T>
void memcpy(HostPointer<T> dst, DevicePointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToHost, hipStreamPerThread)
    );
}

template <typename T>
void memcpy(DevicePointer<T> dst, HostPointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyHostToDevice, hipStreamPerThread)
    );
}

template <typename T>
void memcpy(DevicePointer<T> dst, DevicePointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToDevice, hipStreamPerThread)
    );
}

template <typename T, int N, typename Pointer>
class NDBase {
public:
    NDBase() = default;
    NDBase(std::array<size_t, N> dims) : dims(dims) {}
    NDBase(std::array<size_t, N> dims, T* ptr) : dims(dims), ptr(ptr) {}

    // Copy assignment
    template <typename S>
    NDBase& operator=(const NDBase<T, N, S>& other) {
        shapecheck(*this, other);
        memcpy(this->ptr, other.ptr, this->size() * sizeof(T));
        return (*this);
    }

    __host__ __device__ inline T operator[](size_t i) const { return ptr[i]; }
    __host__ __device__ inline T& operator[](size_t i) { return ptr[i]; }

    __host__ __device__  T* data() { return ptr; }
    __host__ __device__  const T* data() const { return ptr; }

    __host__ __device__ inline const T& front() const { return ptr[0]; }
    __host__ __device__ inline T& front() { return ptr[0]; }

    __host__ __device__ inline const T& back() const { return ptr[size() - 1]; }
    __host__ __device__ inline T& back() { return ptr[size() - 1]; }

    __host__ __device__ inline T* begin() { return ptr; }
    __host__ __device__ inline T* end() { return ptr + this->size(); }

    __host__ __device__ inline const T* begin() const { return ptr; }
    __host__ __device__ inline const T* end() const { return ptr + this->size(); }

    __host__ __device__  size_t size() const {
        size_t sz {1};
        for (const auto dim : dims) sz *= dim;
        return sz;
    }

    size_t size(int i) const { return dims.at(i); }

    auto shape() const { return dims; }

    void zero() {
        HIPCHECK( hipMemsetAsync(this->ptr, 0, this->size() * sizeof(T), hipStreamPerThread) );
    }

    // These friends are required to allow copy assignment between classes with different
    // memory storage locations.
    friend NDBase<T, N, HostPointer<T>>;
    friend NDBase<T, N, DevicePointer<T>>;

protected:
    std::array<size_t, N> dims {};
    Pointer ptr {};
};

template <typename T, typename S, typename R, typename Q, int N>
void shapecheck(const NDBase<T, N, S>& lhs, const NDBase<R, N, Q>& rhs) {
    if (lhs.shape() != rhs.shape()) {
        fmt::println("Incompatible array shapes");
        abort();
    }
}

template <typename T, int N, typename Pointer>
class Span : public NDBase<T, N, Pointer> {
public:
    Span(std::array<size_t, N> dims, T* ptr) : NDBase<T, N, Pointer>(dims, ptr) {}

    Span<T, 1, HostPointer<T>>(std::vector<T> vec) :
        NDBase<T, N, Pointer>({vec.size()}, vec.data()) {}

    // Copy assignment
    template <typename S>
    Span& operator=(const NDBase<T, N, S>& other) {
        NDBase<T, N, Pointer>::operator=(other);
        return (*this);
    }
};

template <typename T, int N>
using HostSpan = Span<T, N, HostPointer<T>>;

template <typename T, int N>
using DeviceSpan = Span<T, N, DevicePointer<T>>;

template <typename T, int N, typename Pointer>
class Array : public NDBase<T, N, Pointer> {
public:
    explicit Array(const std::array<size_t, N> dims) : NDBase<T, N, Pointer>(dims) {
        malloc(this->ptr, this->size() * sizeof(T));
        this->zero();
    }

    // Explicit copy constructor
    template <typename S>
    explicit Array(const NDBase<T, N, S>& other) : Array(other.shape()) { (*this) = other; }

    // Copy assignment
    template <typename S>
    Array& operator=(const NDBase<T, N, S>& other) {
        NDBase<T, N, Pointer>::operator=(other);
        return (*this);
    }

    // Move constructor
    Array(Array<T, N, Pointer>&& other) { (*this) = std::move(other); }

    // Move assignment
    Array& operator=(Array<T, N, Pointer>&& other) {
        using std::swap;
        swap(this->dims, other.dims);
        swap(this->ptr, other.ptr);
        return (*this);
    }

    // Destructor
    ~Array() {
        if (this->ptr) HIPCHECK( hipFreeAsync(this->ptr, hipStreamPerThread) );
    }

    operator Span<T, N, Pointer>() const {
        return Span<T, N, Pointer> {this->dims, this->ptr};
    }
};

template <typename T, int N>
using HostArray = Array<T, N, HostPointer<T>>;

template <typename T, int N>
using DeviceArray = Array<T, N, DevicePointer<T>>;

template <typename T, typename S, int N>
auto& operator+=(NDBase<T, N, HostPointer<T>>& lhs, const NDBase<S, N, HostPointer<S>>& rhs) {
    shapecheck(lhs, rhs);
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] += rhs[i];
    return lhs;
}

template <typename T, typename S, int N>
auto& operator/=(NDBase<T, N, HostPointer<T>>& lhs, const NDBase<S, N, HostPointer<S>>& rhs) {
    shapecheck(lhs, rhs);
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] /= rhs[i];
    return lhs;
}