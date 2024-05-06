#pragma once

#include <cmath>
#include <unordered_map>

#include "gridspec.h"
#include "memory.h"
#include "util.h"

template <typename T>
double kbalpha() {
    static_assert(static_cast<int>(sizeof(T)) == -1);
    return 0;
}

template <>
double kbalpha<float>() { return 4.2; }

template <>
double kbalpha<double>() { return 10; }

std::mutex kblock;

template <typename T>
const auto& kaiserbessel(const GridSpec gridspec, const double alpha = kbalpha<T>()) {
    std::lock_guard l(kblock);

    // Memoise output
    static std::unordered_map<GridSpec, const HostArray<T, 2>> cache;
    if (auto taper = cache.find(gridspec); taper != cache.end()) {
        return std::get<1>(*taper);
    }

    // Create one-dimensional tapers first. The 2D taper is a product of these 1D tapers.
    std::vector<double> xDim(gridspec.Nx);
    std::vector<double> yDim(gridspec.Ny);

    double pi {::pi_v<double>};
    double norm = std::cyl_bessel_i(0, pi * alpha);

    for (auto oneDim : {&xDim, &yDim}) {
        for (size_t i {}; i < oneDim->size(); ++i) {
            double x {static_cast<double>(i) / oneDim->size() - 0.5};
            (*oneDim)[i] = std::cyl_bessel_i(
                0, pi * alpha * std::sqrt(1 - 4 * x * x)
            ) / norm;
        }
    }

    // Create the full taper as the product of the 1D tapers.
    HostArray<T, 2> taper {gridspec.Nx, gridspec.Ny};
    for (size_t i {}; i < taper.size(); ++i) {
        auto [xpx, ypx] = gridspec.linearToGrid(i);
        taper[i] = xDim[xpx] * yDim[ypx];
    }

    cache.insert(std::make_pair(gridspec, std::move(taper)));
    return cache[gridspec];
}