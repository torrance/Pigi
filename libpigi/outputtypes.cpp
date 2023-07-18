#pragma once

#include <complex>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

template <typename T>
struct LinearData {
    // COLUMN MAJOR
    T xx {}, yx {}, xy {}, yy {};

    template <typename S>
    __host__ __device__
    auto& operator*=(const S& c) {
        xx *= c;
        yx *= c;
        xy *= c;
        yy *= c;
        return (*this);
    }

    template <typename S>
    __host__ __device__
    auto& operator*=(const LinearData<S>& other) {
        xx *= other.xx;
        yx *= other.yx;
        xy *= other.xy;
        yy *= other.yy;
        return (*this);
    }

    template <typename S>
    __host__ __device__
    auto& operator+=(const LinearData<S>& other) {
        xx += other.xx;
        yx += other.yx;
        xy += other.xy;
        yy += other.yy;
        return (*this);
    }
};

template<typename T>
using ComplexLinearData = LinearData<std::complex<T>>;

template <typename T>
struct fmt::formatter<LinearData<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const LinearData<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "[{}, {}; {}, {}]", value.xx, value.xy, value.yx, value.yy
        );
    }
};

template <typename T>
struct StokesI {
    std::complex<T> I;

    __host__ __device__ StokesI<T>& operator=(const ComplexLinearData<T> data) {
        I = (T) 0.5 * (data.xx + data.yy);
        return *this;
    }

    template<typename S>
    __host__ __device__
    StokesI<T>& operator*=(const S x) {
        I *= x;
        return *this;
    }

    __host__ __device__
    StokesI<T>& operator +=(const StokesI<T> x) {
        I += x.I;
        return *this;
    }

    template <typename S>
    __host__ __device__
    StokesI<T>& operator /=(const S x) {
        I /= x;
        return *this;
    }
};