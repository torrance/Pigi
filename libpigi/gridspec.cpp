#pragma once

#include <hip/hip_runtime.h>

struct GridSpec {
    size_t Nx;
    size_t Ny;
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
            (lpx - static_cast<long long>(Nx) / 2) * static_cast<S>(scalelm),
            (mpx - static_cast<long long>(Ny) / 2) * static_cast<S>(scalelm)
        );
    }

    template <typename S>
    __host__ __device__
    inline auto linearToUV(const size_t idx) const {
        auto [upx, vpx] = linearToGrid(idx);

        return std::make_tuple(
            (upx - static_cast<long long>(Nx) / 2) * static_cast<S>(scaleuv),
            (vpx - static_cast<long long>(Ny) / 2) * static_cast<S>(scaleuv)
        );
    }

    template <typename S>
    inline auto UVtoGrid(S u, S v) {
        return std::make_tuple(
            static_cast<S>(u / scaleuv + static_cast<long long>(Nx) / 2),
            static_cast<S>(v / scaleuv + static_cast<long long>(Ny) / 2)
        );
    }

    template <typename S>
    inline auto gridToUV(auto upx, auto vpx) {
        return std::make_tuple(
            static_cast<S>((upx - static_cast<long long>(Nx) / 2) * scaleuv),
            static_cast<S>((vpx - static_cast<long long>(Ny) / 2) * scaleuv)
        );
    }
};