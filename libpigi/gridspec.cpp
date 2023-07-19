#pragma once

#include <hip/hip_runtime.h>

struct GridSpec {
    long long Nx;
    long long Ny;
    double scalelm;
    double scaleuv;

    __host__ __device__ inline auto size() const { return Nx * Ny; }

    __host__ __device__
    inline auto linearToGrid(const size_t idx) const {
        // COLUMN MAJOR ordering
        auto xpx { static_cast<long long>(idx % Nx) };
        auto ypx { static_cast<long long>(idx / Nx) };
        return std::make_tuple(xpx, ypx);
    }

    __host__ __device__
    inline auto gridToLinear(const size_t x, const size_t y) const {
        // COLUMN MAJOR ordering
        return x + y * Nx;
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
};