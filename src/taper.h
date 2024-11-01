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

template <typename T>
std::vector<T> kaiserbessel1D(const size_t length) {
    auto alpha = kbalpha<T>();
    T norm = std::cyl_bessel_i(0, ::pi_v<T> * alpha);

    std::vector<T> xs(length);
    for (size_t i {}; i < length; ++i) {
        double x {static_cast<T>(i) / length - 0.5};
        xs[i] = std::cyl_bessel_i(
            0, ::pi_v<T> * alpha * std::sqrt(1 - 4 * x * x)
        ) / norm;
    }

    return xs;
}

template <typename T>
HostArray<T, 2> kaiserbessel2D(const GridSpec gridspec) {
    // Create one-dimensional tapers first. The 2D taper is a product of these 1D tapers.
    auto xDim = kaiserbessel1D<T>(gridspec.Nx);
    auto yDim = kaiserbessel1D<T>(gridspec.Ny);

    // Create the full taper as the product of the 1D tapers.
    HostArray<T, 2> taper {gridspec.Ny, gridspec.Nx};
    for (size_t i {}; i < taper.size(); ++i) {
        auto [xpx, ypx] = gridspec.linearToGrid(i);
        taper[i] = xDim[xpx] * yDim[ypx];
    }

    return taper;
}

template <typename T>
double psfw_c() {
    static_assert(static_cast<int>(sizeof(T)) == -1);
    return 0;
}

template <>
double psfw_c<float>() { return 0 * ::pi_v<double> / 2; }

template <>
double psfw_c<double>() { return 18 * ::pi_v<double> / 2; }

template <typename T>
std::vector<T> pswf1D(const size_t length) {
    // Hard code the PSFW parameters
    int m = 6;
    int n = 6;
    double c = psfw_c<T>();

    // Precompute the eigenvalue
    double cv;
    std::vector<double> eg(n - m + 2);
    specialfunc::segv(m, n, c, 1, &cv, eg.data());

    std::vector<T> xs(length);
    for (size_t i {}; i < length; ++i) {
        double nu = std::abs(-1 + 2 * i / static_cast<double>(length));
        if (nu >= 1) {
            xs[i] = 0;
            continue;
        }

        double s1f, s1d;
        specialfunc::aswfa(nu, m, n, c, 1, cv, &s1f, &s1d);
        xs[i] = s1f;
    }

    // Normalise
    double N = xs[length / 2];
    for (auto& x : xs) x /= N;

    return xs;
}

template <typename T>
HostArray<T, 2> pswf2D(const GridSpec gridspec) {
    // Create one-dimensional tapers first.
    auto xDim = pswf1D<T>(gridspec.Nx);
    auto yDim = pswf1D<T>(gridspec.Ny);

    // Create the full taper as the product of the 1D tapers.
    HostArray<T, 2> taper {gridspec.Ny, gridspec.Nx};
    for (size_t i {}; i < taper.size(); ++i) {
        auto [xpx, ypx] = gridspec.linearToGrid(i);
        taper[i] = xDim[xpx] * yDim[ypx];
    }

    return taper;
}