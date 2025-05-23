#pragma once

#include <array>
#include <optional>
#include <tuple>

#include <hip/hip_runtime.h>

struct GridSpec {
    long long Nx {};
    long long Ny {};
    double scalel {};
    double scalem {};
    double scaleu {};
    double scalev {};
    double deltal {};
    double deltam {};
    long long deltalpx {};
    long long deltampx {};

    bool operator==(const GridSpec&) const = default;

    static GridSpec fromScaleLM(
        long long Nx, long long Ny, double scalelm, double deltal = 0, double deltam = 0
    ) {
        return GridSpec {
            Nx, Ny, scalelm, scalelm, 1 / (Nx * scalelm), 1 / (Ny * scalelm), deltal, deltam,
            std::llround(deltal / scalelm), std::llround(deltam / scalelm)
        };
    }

    static GridSpec fromScaleUV(
        long long Nx, long long Ny, double scaleu, double scalev, double deltal = 0, double deltam = 0
    ) {
        auto scalel = 1 / (Nx * scaleu);
        auto scalem = 1 / (Ny * scalev);
        return GridSpec {
            Nx, Ny, scalel, scalem, scaleu, scalev, deltal, deltam,
            std::llround(deltal / scalel), std::llround(deltam / scalem)
        };
    }

    __host__ __device__ inline auto size() const {
        return static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    }

    inline std::array<long long, 2> shape() const {
        return {Ny, Nx};  // Row major ordering
    }

    __host__ __device__
    inline auto linearToGrid(const size_t idx) const {
        // Y is the major axis
        auto xpx { static_cast<long long>(idx) % Nx };
        auto ypx { static_cast<long long>(idx) / Nx };
        return std::make_tuple(xpx, ypx);
    }

    inline auto linearToLMpx(const size_t idx) const {
        // Y is the major axis
        auto lpx { static_cast<long long>(idx) % Nx };
        auto mpx { static_cast<long long>(idx) / Nx };

        // Apply offset
        lpx -= deltalpx - Nx / 2;
        mpx += deltampx - Ny / 2;

        return std::make_tuple(lpx, mpx);
    }

    inline std::optional<size_t> LMpxToLinear(long long lpx, long long mpx) const {
        lpx += deltalpx - Nx / 2;
        mpx -= deltampx - Ny / 2;

        if (0 <= lpx && lpx < Nx && 0 <= mpx && mpx < Ny) {
            return gridToLinear(lpx, mpx);
        } else {
            return {};
        }
    }

    __host__ __device__
    inline size_t gridToLinear(const long long x, const long long y) const {
        // Y is the major axis
        return static_cast<size_t>(x) +
               static_cast<size_t>(y) * static_cast<size_t>(Nx);
    }

    template <typename S>
    __host__ __device__
    inline auto linearToSky(const size_t idx) const {
        auto [lpx, mpx] = linearToGrid(idx);
        return gridToLM<S>(lpx, mpx);
    }

    template <typename S>
    __host__ __device__
    inline auto linearToUV(const size_t idx) const {
        auto [upx, vpx] = linearToGrid(idx);
        return gridToUV<S>(upx, vpx);
    }

    template <typename S>
    inline auto UVtoGrid(S u, S v) const {
        return std::make_tuple(
            static_cast<S>(Nx / 2 - u / scaleu),
            static_cast<S>(Ny / 2 + v / scalev)
        );
    }

    template <typename S>
    __host__ __device__
    inline auto gridToUV(auto upx, auto vpx) const {
        return std::make_tuple(
            (Nx / 2 - static_cast<S>(upx)) * static_cast<S>(scaleu),
            (static_cast<S>(vpx) - Ny / 2) * static_cast<S>(scalev)
        );
    }

    template <typename S>
    __host__ __device__
    inline auto gridToLM(auto lpx, auto mpx) const {
        return std::make_tuple(
            static_cast<S>(deltal) - (lpx - Nx / 2) * static_cast<S>(scalel),
            static_cast<S>(deltam) + (mpx - Ny / 2) * static_cast<S>(scalem)
        );
    }
};

template <>
struct std::hash<GridSpec> {
    size_t operator()(const GridSpec& key) const {
        using std::hash;
        return (
            hash<long long>()(key.Nx) ^ hash<long long>()(key.Ny) ^
            hash<double>()(key.scalel) ^ hash<double>()(key.scalem) ^
            hash<double>()(key.scaleu) ^ hash<double>()(key.scalev) ^
            hash<double>()(key.deltal) ^ hash<double>()(key.deltam) ^
            hash<long long>()(key.deltalpx) ^ hash<long long>()(key.deltampx)
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
    double deltal {};
    double deltam {};

    GridSpec grid() const {
        return GridSpec::fromScaleLM(imgNx, imgNy, imgScalelm, deltal, deltam);
    }

    GridSpec padded() const {
        long long paddedNx = imgNx * paddingfactor;
        long long paddedNy = imgNy * paddingfactor;

        // Force even padding >= existing size
        paddedNx = paddedNx + (paddedNx & 1);
        paddedNy = paddedNy + (paddedNy & 1);

        // Upscale to nearest even 5-smooth number
        auto issmooth = [] (auto x) -> bool {
            for (auto p : {2, 3, 5}) {
                while (x % p == 0) x /= p;
            }
            return x == 1;
        };
        while (!issmooth(paddedNx)) paddedNx += 2;
        while (!issmooth(paddedNy)) paddedNy += 2;

        return GridSpec::fromScaleLM(paddedNx, paddedNy, imgScalelm, deltal, deltam);
    }

    GridSpec subgrid() const {
        return GridSpec::fromScaleUV(
            kernelsize, kernelsize, padded().scaleu, padded().scalev, deltal, deltam
        );
    }
};