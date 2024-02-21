#pragma once

#include <array>
#include <tuple>

#include <hip/hip_runtime.h>

struct GridSpec {
    long long Nx;
    long long Ny;
    double scalelm;
    double scaleuv;
    double deltal {};
    double deltam {};

    bool operator==(const GridSpec& other) const {
        return (
            Nx == other.Nx && Ny == other.Ny &&
            scalelm == other.scalelm && scaleuv == other.scaleuv &&
            deltal == other.deltal && deltam == other.deltam
        );
    }

    static GridSpec fromScaleLM(long long Nx, long long Ny, double scalelm, double deltal = 0, double deltam = 0) {
        return GridSpec {Nx, Ny, scalelm, 1 / (Nx * scalelm), deltal, deltam};
    }

    static GridSpec fromScaleUV(long long Nx, long long Ny, double scaleuv, double deltal = 0, double deltam = 0) {
        return GridSpec {Nx, Ny, 1 / (Nx * scaleuv), scaleuv, deltal, deltam};
    }

    __host__ __device__ inline auto size() const {
        return static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    }

    inline std::array<long long, 2> shape() const {
        return {Nx, Ny};
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
            (lpx - Nx / 2) * static_cast<S>(scalelm) + static_cast<S>(deltal),
            (mpx - Ny / 2) * static_cast<S>(scalelm) + static_cast<S>(deltam)
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

template <>
struct std::hash<GridSpec> {
    size_t operator()(const GridSpec& key) const {
        using std::hash;
        return (
            hash<long long>()(key.Nx) ^ hash<long long>()(key.Ny) ^
            hash<double>()(key.scalelm) ^ hash<double>()(key.scaleuv) ^
            hash<long>()(key.deltal) ^ hash<long>()(key.deltam)
        );
    }
};

struct GridConfig {
    long long imgNx {};
    long long imgNy {};
    double imgScalelm {};
    double paddingfactor {1};
    long long kernelsize {};
    long long kernelpadding {};
    double wstep {1};
    double deltal {};
    double deltam {};

    GridSpec grid() const {
        return GridSpec::fromScaleLM(imgNx, imgNy, imgScalelm, deltal, deltam);
    }

    GridSpec padded() const {
        // Upscale to nearest _even_ value
        return GridSpec::fromScaleLM(
            imgNx * paddingfactor + int(int(imgNx * paddingfactor) % 2 == 1),
            imgNy * paddingfactor + int(int(imgNy * paddingfactor) % 2 == 1),
            imgScalelm, deltal, deltam
        );
    }

    GridSpec subgrid() const {
        return GridSpec::fromScaleUV(
            kernelsize, kernelsize, padded().scaleuv, deltal, deltam
        );
    }
};