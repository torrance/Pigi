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
    inline auto& operator+=(const LinearData<S>& other) {
        xx += other.xx;
        yx += other.yx;
        xy += other.xy;
        yy += other.yy;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& lmul(const LinearData<S>& other) {
        auto tmpxx = other.xx * xx + other.xy * yx;
        auto tmpyx = other.yx * xx + other.yy * yx;
        auto tmpxy = other.xx * xy + other.xy * yy;
        auto tmpyy = other.yx * xy + other.yy * yy;

        xx = tmpxx;
        yx = tmpyx;
        xy = tmpxy;
        yy = tmpyy;

        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& rmul(const LinearData<S>& other) {
        auto tmpxx = xx * other.xx + xy * other.yx;
        auto tmpyx = yx * other.xx + yy * other.yx;
        auto tmpxy = xx * other.xy + xy * other.yy;
        auto tmpyy = yx * other.xy + yy * other.yy;

        xx = tmpxx;
        yx = tmpyx;
        xy = tmpxy;
        yy = tmpyy;

        return *this;
    }

    __host__ __device__
    inline auto& inv() {
        auto f = T{1} / ((xx * yy) - (xy * yx));
        xx *= f;
        yy *= f;
        xy *= -f;
        yx *= -f;

        using std::swap;
        swap(xx, yy);

        return *this;
    }

    __host__ __device__
    inline auto& adjoint() {
        using std::swap;
        swap(xy, yx);

        xx = std::conj(xx);
        yx = std::conj(yx);
        xy = std::conj(xy);
        yy = std::conj(yy);

        return *this;
    }

    __host__ __device__
    static LinearData<T> fromBeam(
        LinearData<T> val,
        LinearData<T> Aleft,
        LinearData<T> Aright
    ) {
        // These are inplace
        Aleft.inv();
        Aright.adjoint().inv();

        // No normalisation performed

        // Finally, apply beam correction
        // (inv(Aleft) * this * inv(Aright)')
        val.lmul(Aleft).rmul(Aright);

        return val;
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
    std::complex<T> I {};

    __host__ __device__ StokesI<T>() = default;

    __host__ __device__
    StokesI<T>(const ComplexLinearData<T> data) : I((T) 0.5 * (data.xx + data.yy)) {}

    template<typename S>
    __host__ __device__
    inline auto& operator*=(const S x) {
        I *= x;
        return *this;
    }

    __host__ __device__
    inline auto& operator +=(const StokesI<T> x) {
        I += x.I;
        return *this;
    }

    template <typename S>
    __host__ __device__
    inline auto& operator /=(const S x) {
        I /= x;
        return *this;
    }

    __host__ __device__
    static StokesI<T> fromBeam(
        ComplexLinearData<T> val,
        ComplexLinearData<T> Aleft,
        ComplexLinearData<T> Aright
    ) {
        // These are inplace
        Aleft.inv();
        Aright.adjoint().inv();

        // Calculate norm
        T norm {};
        for (auto J : {
            ComplexLinearData<T> {1, 0, 0, 0},
            ComplexLinearData<T> {0, 1, 0, 0},
            ComplexLinearData<T> {0, 0, 1, 0},
            ComplexLinearData<T> {0, 0, 0, 1}
        }) {
            // J = invAleft * selector * invAright
            J.lmul(Aleft).rmul(Aright);
            norm += std::abs(J.xx.real() + J.yy.real()) +
                    std::abs(J.xx.imag() + J.yy.imag());
        }
        norm *= 0.5;

        // Finally, apply beam correction and normalize
        // (inv(Aleft) * this * inv(Aright)') / norm
        val.lmul(Aleft).rmul(Aright);
        val /= norm;

        return val;
    }
};