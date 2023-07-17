#pragma once

#include <complex>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

template <typename T>
struct Matrix2x2 {
    // COLUMN MAJOR
    T xx {}, yx {}, xy {}, yy {};

    __host__ __device__
    Matrix2x2() {}

    template <typename S>
    __host__ __device__
    Matrix2x2(Matrix2x2<S>& other) :
        xx({other.xx}), yx({other.yx}), xy({other.xy}), yy({other.yy}) {}

    static int size() { return 4; }
};

template <typename T, typename S>
__host__ __device__
auto operator*(const Matrix2x2<T>& A, const S& c) {
    Matrix2x2< decltype(A.xx * c) > B(A);
    return B *= c;
}

template <typename T, typename S>
__host__ __device__
auto& operator*=(Matrix2x2<T>& A, const S& c) {
    A.xx *= c;
    A.yx *= c;
    A.xy *= c;
    A.yy *= c;
    return A;
}

template <typename T, typename S>
__host__ __device__
auto operator*(const Matrix2x2<T>& A, const Matrix2x2<S>& B) {
    Matrix2x2< decltype(A.xx * B.xx) > C(A);
    return C *= B;
}

template <typename T, typename S>
__host__ __device__
auto& operator*=(Matrix2x2<T>& A, const Matrix2x2<S>& B) {
    A.xx *= B.xx;
    A.yx *= B.yx;
    A.xy *= B.xy;
    A.yy *= B.yy;
    return A;
}

template <typename T, typename S>
__host__ __device__
Matrix2x2<T>& operator+=(Matrix2x2<T>& A, const Matrix2x2<S>& B) {
    A.xx += B.xx;
    A.yx += B.yx;
    A.xy += B.xy;
    A.yy += B.yy;
    return A;
}

template <typename T>
struct fmt::formatter<Matrix2x2<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Matrix2x2<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "[{}, {}; {}, {}]", value.xx, value.xy, value.yx, value.yy
        );
    }
};

template <typename T>
struct LinearData : public Matrix2x2<std::complex<T>> {
    __host__ __device__
    LinearData() : Matrix2x2<std::complex<T>>({}) {}

    __host__ __device__
    LinearData(
        std::complex<T> xx, std::complex<T> yx,
        std::complex<T> xy, std::complex<T> yy
    ) : Matrix2x2<std::complex<T>>({xx, yx, xy, yy}) {}

    __host__ __device__
    LinearData(Matrix2x2<std::complex<T>> val) :
        Matrix2x2<std::complex<T>>({val.xx, val.yx, val.xy, val.yy}) {}
};

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

    static int size() { return 1; }

    __host__ __device__ StokesI<T>& operator=(const LinearData<T> data) {
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