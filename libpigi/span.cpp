#pragma once

#include <array>
#include <iostream>

#include "hip.cpp"

template <int N>
struct Dims {
    std::array<size_t, N> dims;
    size_t size() const;

    template <typename... Ts>
    Dims(Ts... dims) : dims({static_cast<size_t>(dims)...}) {}

    bool operator==(const Dims<N>& other) const {
        return dims == other.dims;
    }
};

template <>
size_t Dims<1>::size() const { return dims[0]; }

template <>
size_t Dims<2>::size() const { return dims[0] * dims[1]; }

template <typename T, int N>
class NdSpan {
public:
    NdSpan(T* ptr, Dims<N> dims) : dims(dims), ptr(ptr) {}

    NdSpan<T, 1>(std::vector<T> vec) : dims(vec.size()), ptr(vec.data()) {}

    T operator[](size_t i) const { return ptr[i]; }
    T& operator[](size_t i) { return ptr[i]; }

    size_t size() const { return dims.size(); }
    size_t size(int i) const { return dims.dims[i]; }
    Dims<N> shape() const { return dims; }

    T* data() const { return ptr; }
    T* begin() const { return ptr; }
    T* end() const { return ptr + size(); }

private:
    Dims<N> dims;
    T* ptr;
};

template <typename T>
using SpanVector = NdSpan<T, 1>;

template <typename T>
using SpanMatrix = NdSpan<T, 2>;

template <typename T, int N>
class GPUArray {
public:
    explicit GPUArray(Dims<N> dims) : dims(dims) {
        HIPCHECK( hipMalloc(&ptr, size() * sizeof(T)) );
        HIPCHECK( hipMemset(ptr, 0, size() * sizeof(T)) );
    }

    explicit GPUArray(NdSpan<T, N> span) : dims(span.shape()) {
        hipPointerAttribute_t attrs {};
        auto err = hipPointerGetAttributes(&attrs, span.data());

        // If the pointer has been allocated by the host
        // it won't have been recorded by the hip runtime
        // and we'll get an Invalid Argument error.
        if (err == hipErrorInvalidValue) {
            attrs.memoryType = hipMemoryTypeHost;
        } else {
            HIPCHECK( err );
        }

        hipMemcpyKind kind {};
        switch (attrs.memoryType) {
            case hipMemoryTypeHost:
                kind = hipMemcpyHostToDevice;
                break;
            case hipMemoryTypeDevice:
                kind = hipMemcpyDeviceToDevice;
                break;
            default:
                fmt::println("Unhandled memory type, file: {}, line: {}", __FILE__, __LINE__);
                abort();
        }

        HIPCHECK( hipMalloc(&ptr, dims.size() * sizeof(T)) );
        HIPCHECK( hipMemcpy(ptr, span.data(), dims.size() * sizeof(T), kind) );
    }

    GPUArray(const GPUArray<T, N>& other) = delete;
    GPUArray(GPUArray<T, N>&& other) = delete;
    void operator=(const GPUArray<T, N>& other) = delete;
    void operator=(GPUArray<T, N>&& other) = delete;

    operator NdSpan<T, N>() const {
        return NdSpan<T, N>(ptr, dims);
    }

    ~GPUArray() {
        HIPCHECK( hipFree(ptr) );
    }

    T* data() const { return ptr; }
    size_t size() const { return dims.size(); }
    size_t size(int i) const { return dims.dims[i]; }
    Dims<N> shape() const { return dims; }

    void zero() {
        HIPCHECK( hipMemset(ptr, 0, dims.size() * sizeof(T)) );
    }

private:
    Dims<N> dims;
    T* ptr {};
};

template <typename T>
using GPUVector = GPUArray<T, 1>;

template <typename T>
using GPUMatrix = GPUArray<T, 2>;

template <typename T, int N>
class Array {
public:
    explicit Array(Dims<N> dims) : dims(dims) {
        HIPCHECK( hipHostMalloc(&ptr, size() * sizeof(T)) );
        HIPCHECK( hipMemset(ptr, 0, size() * sizeof(T)) );
    }

    Array(const Array<T, N>& other) = delete;
    Array(Array<T, N>&& other) {
        std::swap(ptr, other.ptr);
        std::swap(dims, other.dims);
    }
    void operator=(const Array<T, N>& other) = delete;
    void operator=(Array<T, N>&& other) = delete;

    operator NdSpan<T, N>() {
        return NdSpan<T, N>(ptr, dims);
    }

    ~Array() {
        HIPCHECK( hipFree(ptr) );
    }

    Array<T, N>& operator=(const GPUArray<T, N>& other) {
        if (shape() != other.shape()) {
            fmt::println(stderr, "Cannot copy array: incompatible array sizes");
            abort();
        }

        HIPCHECK( hipMemcpyDtoH(ptr, other.data(), other.size() * sizeof(T)) );
        return *this;
    }

    template <typename S>
    Array<T, N>& operator+=(const S& other) {
        if (shape() != other.shape()) {
            fmt::println(stderr, "Cannot add array: incompatible array sizes");
            abort();
        }

        for (size_t i {}; i < size(); ++i) {
            ptr[i] += other.data()[i];
        }

        return *this;
    }

    template <typename S>
    Array<T, N>& operator/=(const S& other) {
        if (shape() != other.shape()) {
            fmt::println(stderr, "Cannot add array: incompatible array sizes");
            abort();
        }

        for (size_t i {}; i < size(); ++i) {
            ptr[i] /= other.data()[i];
        }
        
        return *this;
    }

    T* data() const { return ptr; }
    size_t size() const { return dims.size(); }
    size_t size(int i) const { return dims.dims[i]; }
    Dims<N> shape() const { return dims; }

private:
    Dims<N> dims;
    T* ptr {};
};

template <typename T>
using Vector = Array<T, 1>;

template <typename T>
using Matrix = Array<T, 2>;
