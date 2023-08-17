#pragma once

#include <array>
#include <algorithm>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

#include "hip.h"

enum class MemoryStorage { Host, Device };

template <typename T, MemoryStorage M>
class BasePointer {
public:
    __host__ __device__ BasePointer() = default;
    __host__ __device__ BasePointer(const T* ptr) : ptr(const_cast<T*>(ptr)) {}

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

template <typename T>
void memset(HostPointer<T> ptr, int val, size_t sz) {
    memset(static_cast<void*>(ptr), val, sz);
}

template <typename T>
void memset(DevicePointer<T> ptr, int val, size_t sz) {
    HIPCHECK( hipMemsetAsync(ptr, val, sz, hipStreamPerThread) );
}

template <typename T, int N, typename Pointer>
class NDBase {
public:
    NDBase() = default;
    NDBase(std::array<long long, N> dims) : dims(dims) {}
    NDBase(std::array<long long, N> dims, T* ptr) : dims(dims), ptr(ptr) {}

    // Copy constructor
    NDBase(const NDBase& other) = default;

    // Copy assignment
    NDBase& operator=(const NDBase& other) {
        shapecheck(*this, other);
        memcpy(this->ptr, other.ptr, this->size() * sizeof(T));
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return (*this);
    }

    // Copy assignment from other pointer type
    template <typename S>
    NDBase& operator=(const NDBase<T, N, S>& other) {
        shapecheck(*this, other);
        memcpy(this->ptr, other.ptr, this->size() * sizeof(T));
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return (*this);
    }

    NDBase(NDBase&& other) = delete;
    NDBase& operator=(NDBase&& other) = delete;

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
        for (const auto dim : dims) sz *= static_cast<size_t>(dim);
        return sz;
    }

    auto size(int i) const { return dims.at(i); }

    auto shape() const { return dims; }

    void zero() {
        memset(this->ptr, 0, this->size() * sizeof(T));
    }

    void fill(const T& val) requires(std::is_same<HostPointer<T>, Pointer>::value) {
        for (auto& x : (*this)) {
            x = val;
        }
    }

    template <typename F, typename... Ss>
    void mapInto(F f, Ss... ins) requires(
        std::is_same<DevicePointer<T>, Pointer>::value
    ) {
        // Each input array must have the same dimensions as the output
        (shapecheck(*this, ins), ...);

        auto fn = ::mapInto<F, NDBase<T, N, Pointer>, Ss...>;
        auto [nblocks, nthreads] = getKernelConfig(fn, this->size());
        hipLaunchKernelGGL(fn, nblocks, nthreads, 0, hipStreamPerThread, f, *this, ins...);
    }

    // These friends are required to allow copy assignment between classes with different
    // memory storage locations.
    friend NDBase<T, N, HostPointer<T>>;
    friend NDBase<T, N, DevicePointer<T>>;

protected:
    std::array<long long, N> dims {};
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
    explicit Span(std::array<long long, N> dims, T* ptr) : NDBase<T, N, Pointer>(dims, ptr) {}

    Span(std::vector<T>& vec) requires(N == 1 && std::is_same<HostPointer<T>, Pointer>::value) :
        NDBase<T, N, Pointer>({static_cast<long long>(vec.size())}, vec.data())  {}

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
    explicit Array() = default;

    explicit Array(const std::array<long long, N> dims) : NDBase<T, N, Pointer>(dims) {
        malloc(this->ptr, this->size() * sizeof(T));
        this->zero();
    }

    // Explicit copy constructor
    explicit Array(const Array& other) : Array(other.shape()) {
        NDBase<T, N, Pointer>::operator=(other);
    }

    // Explicit copy constructor from other pointer type
    template <typename S>
    explicit Array(const NDBase<T, N, S>& other) : Array(other.shape()) { (*this) = other; }

    // Copy assignment from other pointer type
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

    auto asSpan() {
        return Span<T, N, Pointer> {this->dims, this->ptr};
    }

    auto asSpan() const {
        return Span<T, N, Pointer> {this->dims, this->ptr};
    }

    operator Span<T, N, Pointer>() const {
        return Span<T, N, Pointer> {this->dims, this->ptr};
    }

    static auto fromVector(const std::vector<T>& other) requires(N == 1) {
        Array<T, N, Pointer> arr({static_cast<long long>(other.size())});
        memcpy(arr.ptr, HostPointer<T> {other.data()}, arr.size() * sizeof(T));
        return arr;
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