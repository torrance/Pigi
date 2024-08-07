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
    __host__ __device__
    explicit operator LinearData<S>() const {
        return LinearData<S> {
            static_cast<S>(xx), static_cast<S>(yx),
            static_cast<S>(xy), static_cast<S>(yy)
        };
    }

    // Conversion to thrust::complex: only used for PSF construction
    __host__ __device__
    explicit operator thrust::complex<float>() const {
        return (xx + yx + xy + yy) / 4;
    }
    __host__ __device__
    explicit operator thrust::complex<double>() const {
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

    // Frobenius norm
    __host__ __device__
    inline auto norm() const{
        return sqrt(
            xx.real() * xx.real() + xx.imag() * xx.imag() +
            yx.real() * yx.real() + yx.imag() * yx.imag() +
            xy.real() * xy.real() + xy.imag() * xy.imag() +
            yy.real() * yy.real() + yy.imag() * yy.imag()
        );
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

#if !defined(__CUDA_ARCH__)
template<typename T, typename S>
__host__
inline auto matmul(const LinearData<T>& lhs, const LinearData<S>& rhs) {
    using R = decltype(lhs.xx * lhs.yy);
    return LinearData<R> {
        lhs.xx * rhs.xx + lhs.xy * rhs.yx,
        lhs.yx * rhs.xx + lhs.yy * rhs.yx,
        lhs.xx * rhs.xy + lhs.xy * rhs.yy,
        lhs.yx * rhs.xy + lhs.yy * rhs.yy
    };
}
#endif

#if defined(__clang__) || defined(__CUDA_ARCH__)
template<typename T, typename S>
__device__
inline auto matmul(const LinearData<T>& lhs, const LinearData<S>& rhs) {
    using R = decltype(lhs.xx * lhs.yy);
    LinearData<R> res;

    cmac(res.xx, lhs.xx, rhs.xx);
    cmac(res.xx, lhs.xy, rhs.yx);
    cmac(res.yx, lhs.yx, rhs.xx);
    cmac(res.yx, lhs.yy, rhs.yx);
    cmac(res.xy, lhs.xx, rhs.xy);
    cmac(res.xy, lhs.xy, rhs.yy);
    cmac(res.yy, lhs.yx, rhs.xy);
    cmac(res.yy, lhs.yy, rhs.yy);

    return res;
}
#endif

template <typename P>
__device__
inline void atomicAdd(LinearData<P>* const x, const LinearData<P>& y) {
    atomicAdd(&x->xx, y.xx);
    atomicAdd(&x->yx, y.yx);
    atomicAdd(&x->xy, y.xy);
    atomicAdd(&x->yy, y.yy);
}

template <typename P>
__device__
inline void atomicSub(LinearData<P>* const x, const LinearData<P>& y) {
    atomicAdd(&x->xx, -y.xx);
    atomicAdd(&x->yx, -y.yx);
    atomicAdd(&x->xy, -y.xy);
    atomicAdd(&x->yy, -y.yy);
}

template <typename T>
struct StokesI {
    thrust::complex<T> I {};

    StokesI() = default;

    template <typename S>
    __host__ __device__ StokesI(const S& I) : I(I) {}

    template <typename S>
    __host__ __device__
    explicit StokesI(const StokesI<S>& stokesI) : I(stokesI.I) {}

    template <typename S>
    __host__ __device__
    explicit StokesI(const LinearData<S>& data) : I(T(0.5) * (data.xx + data.yy)) {}

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
        return StokesI<T>(1 / matmul(Aleft.inv(), Aright.inv().adjoint()).norm());
    }
};

template <typename P>
__host__ __device__
P abs(StokesI<P> x) { return thrust::abs(x.I); }

template <typename P>
__device__
inline void atomicAdd(StokesI<P>* const x, const StokesI<P>& y) {
    atomicAdd(&x->I, y.I);
}