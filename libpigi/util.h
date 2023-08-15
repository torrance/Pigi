#pragma once

#include <complex>
#include <limits>
#include <numbers>
#include <type_traits>

#include <fmt/format.h>
#include <hip/hip_runtime.h>

#include "gridspec.h"

__device__ void sincospif(float, float*, float*);  // Supress IDE warning about device function
__device__ void sincospi(double, double*, double*); // Suppress IDE warning about device function

#ifdef __HIPCC__
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
#else
template <typename T>
__host__ inline auto cispi(const T& theta) {
    auto pi = std::numbers::pi_v<T>;
    return std::complex {
        std::cos(theta * pi), std::sin(theta * pi)
    };
}
#endif


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
inline T deg2rad(const T& x) { return x * std::numbers::pi_v<T> / 180; }

template <typename T> requires(std::is_floating_point<T>::value)
inline T rad2deg(const T& x) { return x * 180 / std::numbers::pi_v<T>; }

template <typename R>
concept Dereferencable = requires(R a) { *a; true; };

template <typename R>
concept Incrementable = requires(R a) { ++a; };

template <typename T, typename... Ts>
class Zip {
public:
    template <typename... Ss>
    class Iterator {
    public:
        Iterator(Ss&&... current) : current(std::make_tuple(current...)) {}

        auto& operator++() requires((Incrementable<Ss> && ...)) {
            std::apply([] (auto&... current) {
                (++current, ...);
            }, current);
            return *this;
        }

        auto operator*() requires((Dereferencable<Ss> && ...)) {
            return std::apply([this] (auto&... current) {
                return forwardtuple(*current...);
            }, current);
        }

        // It is no longer true that begin() and end() types are the same, so this !=
        // operator must work with different Iterator types.
        template <typename... Rs>
        auto operator!=(const Iterator<Rs...>& other) const {
            return allnotequal(
                current, other.getCurrent(), std::index_sequence_for<Ss...>{}
            );
        }

        const auto& getCurrent() const { return current; }

    private:
        std::tuple<Ss...> current;

        template <typename... Qs, size_t... Ns> requires(sizeof...(Qs) == sizeof...(Ss))
        bool allnotequal(
            const std::tuple<Ss...>& lhs,
            const std::tuple<Qs...>& rhs,
            const std::index_sequence<Ns...>
        ) const {
            return (
                true && ... && (std::get<Ns>(lhs) != std::get<Ns>(rhs))
            );
        }

        template <typename... Rs>
        auto forwardtuple(Rs&&... args) const {
            return std::tuple<Rs...>{std::forward<Rs>(args)...};
        }
    };

    template <typename S, typename... Ss>
    Zip(S&& arg, Ss&&... args) : iters{
        std::forward<S>(arg), std::forward<Ss>(args)...
    } {}

    auto begin() {
        return std::apply([] (auto&&... iters) {
            return Iterator(iters.begin()...);
        }, iters);
    }

    auto end() {
        return std::apply([] (auto&&... iters) {
            return Iterator(iters.end()...);
        }, iters);
    }

private:
    std::tuple<T, Ts...> iters;
};

template <typename... Ts>
auto zip(Ts&&... args) {
    return Zip<Ts...>(std::forward<Ts>(args)...);
}

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