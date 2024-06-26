#pragma once

#include <cmath>
#include <unordered_map>

#include "gridspec.h"
#include "memory.h"
#include "specialfunc.h"
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

std::mutex taperlock;

template <typename T>
const auto& kaiserbessel(const GridSpec gridspec, const double alpha = kbalpha<T>()) {
    std::lock_guard l(taperlock);

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

template <typename T>
double psfw_c() {
    static_assert(static_cast<int>(sizeof(T)) == -1);
    return 0;
}

template <>
double psfw_c<float>() { return 4 * ::pi_v<double> / 2; }

template <>
double psfw_c<double>() { return 20 * ::pi_v<double> / 2; }

template <typename T>
const auto& pswf(const GridSpec gridspec) {
    std::lock_guard l(taperlock);

    // Memoise output
    static std::unordered_map<GridSpec, const HostArray<T, 2>> cache;
    if (auto taper = cache.find(gridspec); taper != cache.end()) {
        return std::get<1>(*taper);
    }

    // Create one-dimensional tapers first. The 2D taper is a product of these 1D tapers.
    std::vector<double> xDim(gridspec.Nx);
    std::vector<double> yDim(gridspec.Ny);

    // Hard code the PSFW parameters
    int m = 6;
    int n = 6;
    double c = psfw_c<T>();

    // Precompute the eigenvalue
    double cv;
    std::vector<double> eg(n - m + 2);
    specialfunc::segv(m, n, c, 1, &cv, eg.data());

    double s1f, s1d;

    for (auto oneDim : {&xDim, &yDim}) {
        for (size_t i {}; i < oneDim->size(); ++i) {
            double nu = std::abs(-1 + 2 * i / static_cast<double>(oneDim->size()));
            if (nu >= 1) {
                (*oneDim)[i] = 0;
                continue;
            }
            specialfunc::aswfa(nu, m, n, c, 1, cv, &s1f, &s1d);
            (*oneDim)[i] = s1f;
        }
    }

    // Normalise
    for (auto oneDim : {&xDim, &yDim}) {
        double N = (*oneDim)[oneDim->size() / 2];
        for (auto& x : *oneDim) x /= N;
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