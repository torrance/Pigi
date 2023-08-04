#pragma once

#include <complex>
#include <numbers>
#include <type_traits>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

void sincospif(float, float*, float*);  // Supress IDE warning about device function

__device__ inline auto cispi(float theta) {
    float real, imag;
    sincospif(theta, &imag, &real);
    return std::complex(real, imag);
}

void sincospi(double, double*, double*); // Suppress IDE warning about device function

__device__ inline auto cispi(double theta) {
    double real, imag;
    sincospi(theta, &imag, &real);
    return std::complex(real, imag);
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
    auto r2 = min(
        l*l + m*m, 1
    );
    return r2 / (1 + sqrt(1 - r2));
}