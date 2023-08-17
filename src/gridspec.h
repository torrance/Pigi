#pragma once

#include <hip/hip_runtime.h>

struct GridSpec {
    long long Nx;
    long long Ny;
    double scalelm;
    double scaleuv;

    static auto fromScaleLM(long long Nx, long long Ny, double scalelm) {
        return GridSpec {Nx, Ny, scalelm, 1 / (Nx * scalelm)};
    }

    static auto fromScaleUV(long long Nx, long long Ny, double scaleuv) {
        return GridSpec {Nx, Ny, 1 / (Nx * scaleuv), scaleuv};
    }

    __host__ __device__ inline auto size() const {
        return static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    }

    __host__ __device__
    inline auto linearToGrid(const size_t idx) const {
        // COLUMN MAJOR ordering
        auto xpx { static_cast<long long>(idx) % Nx };
        auto ypx { static_cast<long long>(idx) / Nx };
        return std::make_tuple(xpx, ypx);
    }

    __host__ __device__
    inline auto gridToLinear(const long long x, const long long y) const {
        // COLUMN MAJOR ordering
        return static_cast<size_t>(x) +
               static_cast<size_t>(y) * static_cast<size_t>(Nx);
    }

    template <typename S>
    __host__ __device__
    inline auto linearToSky(const size_t idx) const {
        auto [lpx, mpx] = linearToGrid(idx);

        return std::make_tuple(
            (lpx - Nx / 2) * static_cast<S>(scalelm),
            (mpx - Ny / 2) * static_cast<S>(scalelm)
        );
    }

    template <typename S>
    __host__ __device__
    inline auto linearToUV(const size_t idx) const {
        auto [upx, vpx] = linearToGrid(idx);

        return std::make_tuple(
            (upx - Nx / 2) * static_cast<S>(scaleuv),
            (vpx - Ny / 2) * static_cast<S>(scaleuv)
        );
    }

    template <typename S>
    inline auto UVtoGrid(S u, S v) const {
        return std::make_tuple(
            static_cast<S>(u / scaleuv + Nx / 2),
            static_cast<S>(v / scaleuv + Ny / 2)
        );
    }

    template <typename S>
    inline auto gridToUV(auto upx, auto vpx) const {
        return std::make_tuple(
            static_cast<S>((upx - Nx / 2) * scaleuv),
            static_cast<S>((vpx - Ny / 2) * scaleuv)
        );
    }
};