#pragma once

#include <array>
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "hip.cpp"

template <int N>
class Dims {
public:
    Dims(std::array<size_t, N> dims) : dims(dims) {}

    Dims<1>(size_t a) : dims({a}) {}
    Dims<1>(long long a) : dims({static_cast<size_t>(a)}) {}

    Dims<2>(size_t a, size_t b) : dims({a, b}) {}
    Dims<2>(long long a, long long b) : dims({static_cast<size_t>(a), static_cast<size_t>(b)}) {}

    bool operator==(const Dims<N>& other) const {
        return dims == other.dims;
    }

    __host__ __device__ inline size_t size() const;
    size_t size(int i) const { return dims.at(i); }
    auto shape() const { return dims; }

private:
    std::array<size_t, N> dims {};
};

template<> __host__ __device__ inline size_t Dims<1>::size() const { return dims[0]; }
template<> __host__ __device__ inline size_t Dims<2>::size() const { return dims[0] * dims[1]; }

template <typename T, int N>
class NdSpan : public Dims<N> {
public:
    NdSpan(T* ptr, Dims<N> dims) : Dims<N>(dims), ptr(ptr) {}

    NdSpan<T, 1>(std::vector<T> vec) : Dims<1>(vec.size()), ptr(vec.data()) {}
    NdSpan<T, 1>(T* start, T* end) : Dims<1>(end - start), ptr(start) {}

    __host__ __device__ inline T operator[](size_t i) const { return ptr[i]; }
    __host__ __device__ inline T& operator[](size_t i) { return ptr[i]; }

    __host__ __device__ inline T* data() const { return ptr; }
    __host__ __device__ inline T* begin() const { return ptr; }
    __host__ __device__ inline T* end() const { return ptr + this->size(); }

private:
    T* __restrict__ ptr;
};

template <typename T>
using SpanVector = NdSpan<T, 1>;

template <typename T>
using SpanMatrix = NdSpan<T, 2>;

template <typename T, int N>
class DeviceArray : public Dims<N> {
public:
    explicit DeviceArray(Dims<N> dims) : Dims<N>(dims) {
        HIPCHECK( hipMallocAsync(&ptr, this->size() * sizeof(T), hipStreamPerThread) );
        HIPCHECK( hipMemsetAsync(ptr, 0, this->size() * sizeof(T), hipStreamPerThread) );
    }

    explicit DeviceArray(NdSpan<T, N> span) : Dims<N>(span.shape()) {
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

        HIPCHECK( hipMallocAsync(&ptr, this->size() * sizeof(T), hipStreamPerThread) );
        HIPCHECK( hipMemcpyAsync(ptr, span.data(), this->size() * sizeof(T), kind, hipStreamPerThread) );
    }

    DeviceArray(const DeviceArray<T, N>& other) = delete;
    void operator=(const DeviceArray<T, N>& other) = delete;
    void operator=(DeviceArray<T, N>&& other) = delete;

    DeviceArray(DeviceArray<T, N>&& other) = default;

    operator NdSpan<T, N>() const {
        return NdSpan<T, N>(ptr, this->shape());
    }

    ~DeviceArray() {
        HIPCHECK( hipFreeAsync(ptr, hipStreamPerThread) );
    }

    T* data() const { return ptr; }

    void zero() {
        HIPCHECK( hipMemsetAsync(ptr, 0, this->size() * sizeof(T), hipStreamPerThread) );
    }

private:
    T* ptr {};
};

template <typename T>
using DeviceVector = DeviceArray<T, 1>;

template <typename T>
using DeviceMatrix = DeviceArray<T, 2>;

template <typename T, int N>
class HostArray : public Dims<N> {
public:
    explicit HostArray(Dims<N> dims) : Dims<N>(dims) {
        HIPCHECK( hipHostMalloc(&ptr, this->size() * sizeof(T)) );
        HIPCHECK( hipMemset(ptr, 0, this->size() * sizeof(T)) );
    }

    HostArray(const HostArray<T, N>& other) = delete;
    void operator=(const HostArray<T, N>& other) = delete;
    void operator=(HostArray<T, N>&& other) = delete;

    HostArray(HostArray<T, N>&& other) = default;

    operator NdSpan<T, N>() {
        return NdSpan<T, N>(ptr, this->dims);
    }

    ~HostArray() {
        HIPCHECK( hipFree(ptr) );
    }

    HostArray<T, N>& operator=(const DeviceArray<T, N>& other) {
        if (this->shape() != other.shape()) {
            fmt::println(stderr, "Cannot copy array: incompatible array sizes");
            abort();
        }

        HIPCHECK( hipMemcpyDtoH(ptr, other.data(), other.size() * sizeof(T)) );
        return *this;
    }

    template <typename S>
    HostArray<T, N>& operator+=(const S& other) {
        if (this->shape() != other.shape()) {
            fmt::println(stderr, "Cannot add array: incompatible array sizes");
            abort();
        }

        for (size_t i {}; i < this->size(); ++i) {
            ptr[i] += other.data()[i];
        }

        return *this;
    }

    template <typename S>
    HostArray<T, N>& operator/=(const S& other) {
        if (this->shape() != other.shape()) {
            fmt::println(stderr, "Cannot add array: incompatible array sizes");
            abort();
        }

        for (size_t i {}; i < this->size(); ++i) {
            ptr[i] /= other.data()[i];
        }
        
        return *this;
    }

    T* data() const { return ptr; }

private:
    T* ptr {};
};

template <typename T>
using HostVector = HostArray<T, 1>;

template <typename T>
using HostMatrix = HostArray<T, 2>;
