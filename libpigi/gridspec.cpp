#pragma once

#include <hip/hip_runtime.h>

struct GridSpec {
    long long Nx;
    long long Ny;
    double scalelm;
    double scaleuv;
};

__host__ __device__
inline auto linearToXY(size_t idx, GridSpec gridspec) {
    // COLUMN MAJOR ordering
    auto xpx { static_cast<long long>(idx % gridspec.Nx) };
    auto ypx { static_cast<long long>(idx / gridspec.Nx) };
    return std::make_tuple(xpx, ypx);
}

__host__ __device__
inline auto XYToLinear(size_t x, size_t y, GridSpec gridspec) {
    // COLUMN MAJOR ordering
    return y * gridspec.Nx + x;
}