#pragma once

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include "util.h"

template <typename T>
struct alignas(16) LinearData {
    // COLUMN MAJOR
    T xx {}, yx {}, xy {}, yy {};

    template <typename S>
    explicit operator LinearData<S>() const {
        return LinearData<S> {
            static_cast<S>(xx), static_cast<S>(yx),
            static_cast<S>(xy), static_cast<S>(yy)
        };
    }

    // Conversion to thrust::complex: only used for PSF construction
    __host__ __device__
    explicit operator thrust::complex<float>() requires(std::is_same<T, thrust::complex<float>>::value) {
        return (xx + yx + xy + yy) / 4;
    }
    __host__ __device__
    explicit operator thrust::complex<double>() requires(std::is_same<T, thrust::complex<double>>::value) {
        return (xx + yx + xy + yy) / 4;
    }

    bool operator==(const LinearData& other) const {
        return xx == other.xx && yx == other.yx && xy == other.xy && yy == other.yy;
    }

    bool operator!=(const LinearData& other) const { return !(*this == other); };

    template <typename S>
    __host__ __device__
    inline auto& operator*=(const S& c) {
        xx *= c;
        yx *= c;
        xy *= c;
        yy *= c;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator/=(const S& c) {
        xx /= c;
        yx /= c;
        xy /= c;
        yy /= c;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator*=(const LinearData<S>& other) {
        xx *= other.xx;
        yx *= other.yx;
        xy *= other.xy;
        yy *= other.yy;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator/=(const LinearData<S>& other) {
        xx /= other.xx;
        yx /= other.yx;
        xy /= other.xy;
        yy /= other.yy;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator+=(const LinearData<S>& other) {
        xx += other.xx;
        yx += other.yx;
        xy += other.xy;
        yy += other.yy;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator-=(const LinearData<S>& other) {
        xx -= other.xx;
        yx -= other.yx;
        xy -= other.xy;
        yy -= other.yy;
        return *this;
    }

    __host__ __device__
    inline LinearData<T> inv() const {
        auto f = T{1} / ((xx * yy) - (xy * yx));
        return {f * yy, -f * yx, -f * xy, f * xx};
    }

    __host__ __device__
    inline LinearData<T> adjoint() const {
        return {::conj(xx), ::conj(xy), ::conj(yx), ::conj(yy)};
    }

    inline auto isfinite() const {
        return ::isfinite(xx) && ::isfinite(yx) && ::isfinite(yx) && ::isfinite(yy);
    }

    auto begin() { return &xx;}
    auto end() { return &yy; }

    auto operator[](size_t i) const { return begin()[i]; }
    auto& operator[](size_t i) { return begin()[i]; }
};

template<typename T>
using ComplexLinearData = LinearData<thrust::complex<T>>;

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

template<typename T, typename S>
__host__ __device__
inline auto matmul(const LinearData<T>& lhs, const LinearData<S>& rhs) {
    using R = decltype(lhs.xx * lhs.yy);
    return LinearData<R> {
        lhs.xx * rhs.xx + lhs.xy * rhs.yx,
        lhs.yx * rhs.xx + lhs.yy * rhs.yx,
        lhs.xx * rhs.xy + lhs.xy * rhs.yy,
        lhs.yx * rhs.xy + lhs.yy * rhs.yy
    };
}

template <typename T>
struct StokesI {
    thrust::complex<T> I {};

    StokesI() = default;

    template <typename S>
    __host__ __device__ StokesI(const S& I) : I{I} {}

    __host__ __device__
    explicit StokesI(const ComplexLinearData<T>& data) : I((T) 0.5 * (data.xx + data.yy)) {}

    __host__ __device__
    operator ComplexLinearData<T>() const {
        return ComplexLinearData<T> {I, 0, 0, I};
    }

    __host__ __device__
    inline auto& operator +=(const StokesI<T>& x) {
        I += x.I;
        return *this;
    }

    __host__ __device__
    inline auto& operator -=(const StokesI<T>& x) {
        I -= x.I;
        return *this;
    }

    __host__ __device__
    inline auto& operator *=(const StokesI<T>& x) {
        I *= x.I;
        return *this;
    }

    __host__ __device__
    inline auto& operator /=(const StokesI<T>& x) {
        I /= x.I;
        return *this;
    }

    template<typename S>
    __host__ __device__
    inline auto& operator*=(const S x) {
        I *= x;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator /=(const S x) {
        I /= x;
        return *this;
    }

    static StokesI<T> beamPower(ComplexLinearData<T>& Aleft, ComplexLinearData<T>& Aright) {
        StokesI<T> norm = static_cast<StokesI<T>>(
            matmul(
                matmul(Aleft.inv(), ComplexLinearData<T> {{1, 1}, {1, 1}, {1, 1}, {1, 1}}),
                Aright.inv().adjoint()
            )
        );
        return 1 / thrust::abs(norm.I);
    }
};

template <typename P>
__host__ __device__
P abs(StokesI<P> x) { return thrust::abs(x.I); }