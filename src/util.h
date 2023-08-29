#pragma once

#include <complex>
#include <limits>
#include <type_traits>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

#include "gridspec.h"
#include "memory.h"

// This is a std::numbers polyfill
// TODO: remove when we can use a more up to date standard library
template <typename T>
constexpr T pi_v = static_cast<T>(3.14159265358979323846264338327950288419716939937510L);

__device__ void sincospif(float, float*, float*);  // Supress IDE warning about device function
__device__ void sincospi(double, double*, double*);  // Suppress IDE warning about device function

__device__ inline auto cispi(const float& theta) {
    float real, imag;
    sincospif(theta, &imag, &real);
    return std::complex(real, imag);
}

__device__ inline auto cispi(const double& theta) {
    double real, imag;
    sincospi(theta, &imag, &real);
    return std::complex(real, imag);
}

template <typename T>
__host__ inline auto cispi(const T& theta) {
    auto pi = ::pi_v<T>;
    return std::complex {
        std::cos(theta * pi), std::sin(theta * pi)
    };
}


template <typename T>
struct fmt::formatter<std::complex<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::complex<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "{:.2f} + {:.2f}i", value.real(), value.imag()
        );
    }
};

template <typename T>
inline bool isfinite(const T& x) requires(std::is_floating_point<T>::value) {
    return std::isfinite(x);
}

template <typename T>
inline bool isfinite(const std::complex<T>& x) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
}

template <typename T>
__host__ __device__
inline T conj(const T& x) requires(std::is_floating_point<T>::value) {
    return x;
}

template <typename T>
__host__ __device__
inline std::complex<T> conj(const std::complex<T>& x) { return std::conj(x); }

template <typename T>
__host__ __device__
inline T ndash(const T l, const T m) {
    auto r2 = std::min<T>(
        l*l + m*m, 1
    );
    return r2 / (1 + sqrt(1 - r2));
}

template <typename T> requires(std::is_floating_point<T>::value)
inline T deg2rad(const T& x) { return x * ::pi_v<T> / 180; }

template <typename T> requires(std::is_floating_point<T>::value)
inline T rad2deg(const T& x) { return x * 180 / ::pi_v<T>; }

template <typename T>
auto crop(HostArray<T, 2>& img, long long edgex, long long edgey) {
    HostArray<T, 2> cropped({
        std::max(img.size(0) - 2 * edgex, 0ll),
        std::max(img.size(1) - 2 * edgey, 0ll)
    });

    GridSpec gridspecSrc {img.size(0), img.size(1), 0, 0};
    GridSpec gridspecDst {cropped.size(0), cropped.size(1), 0, 0};

    for (long long nx {}; nx < cropped.size(0); ++nx) {
        for (long long ny {}; ny < cropped.size(1); ++ny) {
            auto idxSrc = gridspecSrc.gridToLinear(nx + edgex, ny + edgey);
            auto idxDst = gridspecDst.gridToLinear(nx, ny);
            cropped[idxDst] = img[idxSrc];
        }
    }

    return cropped;
}

template <typename T>
auto crop(HostArray<T, 2>& img, long long edge) {
    return crop(img, edge, edge);
}

template <typename T=size_t>
class Iota {
public:
    class Iterator {
    public:
        Iterator(T val) : val(val) {}

        Iterator& operator++() {
            ++val;
            return *this;
        }

        T operator*() const { return val; }

        bool operator==(const Iterator& other) const { return val == other.val; }
        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        T val;
    };

    __host__ __device__
    T operator[](size_t i) { return i; }

    size_t size() { return std::numeric_limits<T>::max(); }

    Iterator begin() { return Iterator{ 0 }; }
    Iterator end() { return Iterator{ std::numeric_limits<T>::max() }; }
};

template <typename F, typename T, typename... Ts>
__global__
void _map(size_t N, F f, T x, Ts... xs) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        f(x[idx], xs[idx]...);
    }
}

auto _mapforward(const auto& x) { return x; }

template <typename T, int N, typename Pointer>
Span<T, N, Pointer> _mapforward(Array<T, N, Pointer>& x) { return x; }

template <typename T, int N, typename Pointer>
const Span<T, N, Pointer> _mapforward(const Array<T, N, Pointer>& x) { return x; }

template<typename F, typename T, typename... Ts>
void map(F f, T&& x, Ts&&... xs) {
    size_t N { std::min({x.size(), xs.size()...}) };
    auto fn = _map<F, decltype(_mapforward(x)), decltype(_mapforward(xs))...>;
    auto [nblocks, nthreads] = getKernelConfig(fn, N);
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        N, f, _mapforward(x), _mapforward(xs)...
    );
}

/**
 * Ceiling integer division
 */
template <typename T, typename S>
requires(std::is_integral<T>::value && std::is_integral<S>::value)
__host__ __device__
inline auto cld(T x, S y) {
    return (x + y - 1) / y;
}