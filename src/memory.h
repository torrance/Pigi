#pragma once

#include <array>
#include <algorithm>
#include <mutex>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

#include "hip.h"

enum class MemoryStorage { Host, Device };

template <typename T, MemoryStorage M>
class BasePointer {
public:
    BasePointer() = default;
    __host__ __device__ BasePointer(const T* ptr) : ptr(const_cast<T*>(ptr)) {}

    __host__ __device__ inline T& operator*() const { return *ptr; }
    __host__ __device__ inline T** operator&() { return &ptr; }

    __host__ __device__ inline T& operator[](size_t i) const { return ptr[i]; }

    __host__ __device__ inline auto operator+(auto x) const { return BasePointer {ptr + x}; }

    __host__ __device__ inline operator T*() const { return ptr; }
    __host__ __device__ inline operator void*() const {
        return reinterpret_cast<void*>(ptr);
    }
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
    HIPCHECK( hipHostMalloc(reinterpret_cast<void**>(&ptr), sz) );
}

template <typename T>
void malloc(DevicePointer<T>& ptr, size_t sz) {
    HIPCHECK( hipMallocAsync(reinterpret_cast<void**>(&ptr), sz, hipStreamPerThread) );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void free(HostPointer<T>& ptr) {
    HIPCHECK( hipHostFree(ptr) );
}

template <typename T>
void free(DevicePointer<T>& ptr) {
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
    HIPCHECK( hipFreeAsync(ptr, hipStreamPerThread) );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void memcpy(HostPointer<T> dst, HostPointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyHostToHost, hipStreamPerThread)
    );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void memcpy(HostPointer<T> dst, DevicePointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToHost, hipStreamPerThread)
    );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void memcpy(DevicePointer<T> dst, HostPointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyHostToDevice, hipStreamPerThread)
    );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void memcpy(DevicePointer<T> dst, DevicePointer<T> src, size_t sz) {
    HIPCHECK(
        hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToDevice, hipStreamPerThread)
    );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T>
void memset(HostPointer<T> ptr, int ch, size_t count) {
    memset(static_cast<void*>(ptr), ch, count);
}

template <typename T>
void memset(DevicePointer<T> ptr, int ch, size_t count) {
    HIPCHECK(
        hipMemsetAsync(static_cast<void*>(ptr), ch, count, hipStreamPerThread)
    );
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
}

template <typename T, int N, typename Pointer>
class Span {
public:
    Span() = default;
    Span(std::array<long long, N> dims, T* ptr = NULL) : dims(dims), ptr(ptr) {
        // Compute size
        sz = std::apply([] (auto... dims) {
            return (1 * ... * static_cast<size_t>(dims));
        }, dims);
    }

    template <typename Alloc>
    Span(std::vector<T, Alloc>& vec) requires(
        N == 1 && std::is_same<HostPointer<T>, Pointer>::value
    ) : Span<T, N, Pointer>({static_cast<long long>(vec.size())}, vec.data())  {}

    __host__ __device__ inline T operator[](size_t i) const { return ptr[i]; }
    __host__ __device__ inline T& operator[](size_t i) { return ptr[i]; }

    Span<T, N - 1, Pointer> operator()(long long i) requires (N > 1) {
        // Check bounds
        if (i < 0 || i > dims[0]) throw std::out_of_range(fmt::format(
            "Attempting to index {} into Span with outer dimension length {}", i, dims[0]
        ));

        // Create new dims with trucated outer dimension
        std::array<long long, N - 1> newdims {};
        for (int j {}; j < N - 1; ++j) newdims[j] = dims[j + 1];

        // Compute offset based on stride
        size_t offset = i;
        offset = std::apply([&] (auto... dims) {
            return (offset * ... * static_cast<size_t>(dims));
        }, newdims);

        return Span<T, N - 1, Pointer>(newdims, ptr + offset);
    }

    __host__ __device__  T* data() { return ptr; }
    __host__ __device__  const T* data() const { return ptr; }

    __host__ __device__  Pointer pointer() { return ptr; }
    __host__ __device__  const Pointer pointer() const { return ptr; }

    __host__ __device__ inline const T& front() const { return ptr[0]; }
    __host__ __device__ inline T& front() { return ptr[0]; }

    __host__ __device__ inline const T& back() const { return ptr[size() - 1]; }
    __host__ __device__ inline T& back() { return ptr[size() - 1]; }

    __host__ __device__ inline T* begin() { return ptr; }
    __host__ __device__ inline T* end() { return ptr + this->size(); }

    __host__ __device__ inline const T* begin() const { return ptr; }
    __host__ __device__ inline const T* end() const { return ptr + this->size(); }

    __host__ __device__  size_t size() const { return sz; }

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

    // These friends are required to allow copy assignment between classes with different
    // memory storage locations.
    friend Span<T, N, HostPointer<T>>;
    friend Span<T, N, DevicePointer<T>>;

protected:
    std::array<long long, N> dims {};
    Pointer ptr {};
    size_t sz {};
};

template <typename T, int N>
using HostSpan = Span<T, N, HostPointer<T>>;

template <typename T, int N>
using DeviceSpan = Span<T, N, DevicePointer<T>>;

template <typename T, typename S, typename R, typename Q, int N>
void shapecheck(const Span<T, N, S>& lhs, const Span<R, N, Q>& rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error(fmt::format(
            "Incompatible array shapes (lhs {} rhs {})",
            lhs.shape(), rhs.shape()
        ));
    }
}

template <typename T, typename PointerDst, typename PointerSrc, int N>
void copy(Span<T, N, PointerDst>& dst, const Span<T, N, PointerSrc>& src) {
    shapecheck(dst, src);
    memcpy(dst.pointer(), src.pointer(), sizeof(T) * dst.size());
}

template <typename T, typename Alloc, typename PointerSrc, int N>
void copy(std::vector<T, Alloc>& dst, const Span<T, N, PointerSrc>& src) {
    // TOFIX: This const_cast is required since we don't currently
    // initialize a span from a const pointer correctly.
    HostSpan<T, 1> span {const_cast<std::vector<T, Alloc>&>(dst)};
    copy(span, src);
}

template <typename T, typename Alloc, typename PointerSrc, int N>
void copy(Span<T, N, PointerSrc>& dst, const std::vector<T, Alloc>& src) {
    // TOFIX: This const_cast is required since we don't currently
    // initialize a span from a const pointer correctly.
    const HostSpan<T, 1> span {const_cast<std::vector<T, Alloc>&>(src)};
    copy(dst, span);
}

template <typename T, int N, typename Pointer>
class Array : public Span<T, N, Pointer> {
public:
    Array() = default;

    explicit Array(const std::array<long long, N>& dims, const bool zero = true) :
        Span<T, N, Pointer>(dims) {

        malloc(this->ptr, this->size() * sizeof(T));
        if (zero) this->zero();
    }

    explicit Array(long long dim0) requires(N == 1) :
        Array(std::array{dim0}) {}
    explicit Array(long long dim0, long long dim1) requires(N == 2):
        Array(std::array{dim0, dim1}) {}

    template <typename Alloc>
    explicit Array(const std::vector<T, Alloc>& vec) requires(N == 1) :
        Array{{static_cast<long long>(vec.size())}, false} {

        copy(*this, vec);
    }

    // Explicit copy constructor
    explicit Array(const Array& other) : Array{other.shape(), false} {
        copy(*this, other);
    }

    // Explicit copy constructor from other pointer type
    template <typename S>
    explicit Array(const Span<T, N, S>& other) : Array{other.shape(), false} {
        copy(*this, other);
    }

    // Delete copy assignment
    Array operator=(const Array&) = delete;

    // Move constructor
    Array(Array<T, N, Pointer>&& other) noexcept { *this = std::move(other); }

    // Move assignment
    Array& operator=(Array<T, N, Pointer>&& other) noexcept {
        using std::swap;
        swap(this->dims, other.dims);
        swap(this->ptr, other.ptr);
        swap(this->sz, other.sz);
        return (*this);
    }

    // Destructor
    ~Array() {
        if (this->ptr) free(this->ptr);
        this->ptr = 0;
    }

    template <typename R>
    explicit operator Array<R, N, HostPointer<R>>() const requires(
        std::is_same<Pointer, HostPointer<T>>::value
    ) {
        Array<R, N, HostPointer<R>> other {this->shape(), false};
        for (size_t i {}; i < this->size(); ++i) {
            other[i] = static_cast<R>((*this)[i]);
        }
        return other;
    }
};

template <typename T, int N>
using HostArray = Array<T, N, HostPointer<T>>;

template <typename T, int N>
using DeviceArray = Array<T, N, DevicePointer<T>>;

template <typename T, int N>
auto& operator+=(Span<T, N, HostPointer<T>>& lhs, const T rhs) {
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] += rhs;
    return lhs;
}

template <typename T, int N>
auto& operator/=(Span<T, N, HostPointer<T>>& lhs, const T rhs) {
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] /= rhs;
    return lhs;
}

template <typename T, typename S, int N>
auto& operator+=(Span<T, N, HostPointer<T>>& lhs, const Span<S, N, HostPointer<S>>& rhs) {
    shapecheck(lhs, rhs);
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] += rhs[i];
    return lhs;
}

template <typename T, typename S, int N>
auto& operator/=(Span<T, N, HostPointer<T>>& lhs, const Span<S, N, HostPointer<S>>& rhs) {
    shapecheck(lhs, rhs);
    for (size_t i {}; i < lhs.size(); ++i) lhs[i] /= rhs[i];
    return lhs;
}