#pragma once

#include <array>
#include <cmath>
#include <chrono>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>

#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "polyfit.h"

namespace clean {

template <typename T>
using ComponentMap = std::unordered_map<size_t, T>;

template <typename S, int N>
auto _major(
    std::vector<double>& freqs,
    std::vector<HostArray<StokesI<S>, 2>>& residuals,
    const GridSpec imgGridspec,
    std::vector<HostArray<thrust::complex<S>, 2>>& psfs,
    const GridSpec psfGridspec,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter
) {
    std::chrono::nanoseconds maxFindingDuration {};
    std::chrono::nanoseconds psfSubtractionDuration {};

    // Create components maps to return
    // We use maps here since component maps are usually sparse
    std::vector<ComponentMap<StokesI<S>>> components(N);

    // Clean down to either:
    //   1. the autothreshold limit, or
    //   1. the explicit threshold limit, or
    //   2. the current peak value minus the majorgain
    // (whichever is greater).

    // Estimate noise of image combined across all channel groups
    S noise {};
    {
        HostArray<StokesI<S>, 2> imgCombined {imgGridspec.shape()};
        for (auto& residual : residuals) {
            imgCombined += residual;
        }
        imgCombined /= StokesI<S>(N);

        S mean {};
        for (auto& val : imgCombined) { mean += val.I.real(); }
        mean /= imgCombined.size();

        S variance {};
        for (auto& val : imgCombined) { variance += std::pow(val.I.real() - mean, 2); }
        variance /= imgCombined.size();

        noise = std::sqrt(variance);
    }

    // Find initial maxVal
    S maxVal {};
    for (size_t idx {}; idx < imgGridspec.size(); ++idx) {
        S val {};
        for (auto& residual : residuals) {
            val += residual[idx].I.real();
        }
        maxVal = std::max(maxVal, std::abs(val / N));
    }

    // Set threshold as max of the possible methods
    S threshold = (1 - majorgain) * maxVal;
    bool finalMajor {false};

    if (threshold < noise * autoThreshold || threshold < cleanThreshold) {
        threshold = std::max(noise * autoThreshold, cleanThreshold);
        finalMajor = true;
    }

    fmt::println(
        "Beginning{}major clean cycle: from {:.2g} Jy to {:.2g} Jy (est. noise {:.2g} Jy)",
        finalMajor ? " (final) " : " ", maxVal, threshold, noise
    );

    // Reduce search space only to existing pixels above the threshold;
    using Pixel = std::tuple<size_t, std::array<S, N>>;
    std::vector<Pixel> pixels;
    for (size_t i {}; i < imgGridspec.size(); ++i) {
        std::array<S, N> vals;
        for (size_t n {}; n < N; ++n) {
            vals[n] = residuals[n][i].I.real();
        }

        auto meanVal = std::apply([] (auto... vals) {
            return ((vals + ...)) / N;
        }, vals);

        if (std::abs(meanVal) >= threshold) {
            pixels.push_back({i, vals});
        }
    }

    // Combine PSFs so that N axis is dense
    HostArray<std::array<S, N>, 2> psfsDense {psfGridspec.shape()};
    for (size_t n {}; n < N; ++n) {
        auto& psf = psfs[n];
        for (size_t i {}; i < psfGridspec.size(); ++i) {
            psfsDense[i][n] = psf[i].real();
        }
    }

    size_t iter {1};
    for (; iter < niter; ++iter) {
        auto start = std::chrono::steady_clock::now();

        // Find the maximum value
        maxVal = 0;
        Pixel maxPixel {};
        for (auto& pixel : pixels) {
            auto& [_, vals] = pixel;
            auto meanVal = std::apply([] (auto... vals) {
                return std::abs((vals + ...)) / N;
            }, vals);

            if (meanVal > maxVal) {
                maxVal = meanVal;
                maxPixel = pixel;
            }
        }

        auto& [maxIdx, maxVals] = maxPixel;

        // Recalculate mean value with correct sign (for logging)
        maxVal = std::apply([] (auto... vals) { return ((vals + ...)) / N; }, maxVals);

        maxFindingDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start
        );

        // Fit polynomial to output
        // TODO: make root configurable
        auto coeffs = polyfit<double, S>(freqs, maxVals, std::min(N, 2));

        // Evalate polynomial model and apply gain
        for (size_t n {}; n < N; ++n) {
            maxVals[n] = 0;
            for (size_t i {}; i < coeffs.size(); ++i) {
                maxVals[n] += coeffs[i] * std::pow(freqs[n], i);
            }
            maxVals[n] *= static_cast<S>(minorgain);
        }

        start = std::chrono::steady_clock::now();

        // Save component
        for (size_t n {}; n < N; ++n) {
            components[n][maxIdx] += StokesI<S>(maxVals[n]);
        }

        // Find grid locations for max index
        auto [xpeak, ypeak] = imgGridspec.linearToGrid(maxIdx);

        // Subtract (component * psf) from pixels
        for (auto& pixel : pixels) {
            auto& [i, vals] = pixel;

            // Calculate respective xpx, ypx in psf image
            // First, get xy coordinates of pixel wrt img
            auto [xpx, ypx] = imgGridspec.linearToGrid(i);

            // Find offset from peak center
            xpx -= xpeak;
            ypx -= ypeak;

            // Convert psf coordinates wrt to bottom left corner
            xpx += psfGridspec.Nx / 2;
            ypx += psfGridspec.Ny / 2;

            // Subtract component from pixel values if it maps to a valid psf pixel
            if (0 <= xpx && xpx < psfGridspec.Nx && 0 <= ypx && ypx < psfGridspec.Ny) {
                auto idx = psfGridspec.gridToLinear(xpx, ypx);

                auto psf = psfsDense[idx];
                for (size_t n {}; n < N; ++n) {
                    vals[n] -= maxVals[n] * psf[n];
                }
            }
        }

        psfSubtractionDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start
        );

        if (iter % 1000 == 0) {
            // Filter list for any pixels that have now fallen beneath the threshold
            std::erase_if(pixels, [=] (auto& pixel) {
                auto& [_, vals] = pixel;
                auto meanVal = std::apply([] (auto... vals) {
                    return ((vals + ...)) / N;
                }, vals);
                return std::abs(meanVal) < threshold;
            });

            fmt::println(
                "   [{} iteration] {:.3f} Jy peak found; search space {} pixels",
                iter, maxVal, pixels.size()
            );
        }

        if (std::abs(maxVal) <= threshold) break;
    }

    S subtractedFlux {};
    for (auto& componentMap : components) {
        for (auto& [_, val] : componentMap) {
            subtractedFlux += val.I.real();
        }
    }
    subtractedFlux /= N;

    fmt::println(
        "Clean cycle complete ({} iterations this major cycle; {:.2f} s peak finding; {:.2f} s PSF subtraction). Subtracted flux: {:.4g} Jy",
        iter, maxFindingDuration.count() / 1e9,
        psfSubtractionDuration.count() / 1e9, subtractedFlux
    );

    return std::make_tuple(std::move(components), iter, finalMajor);
}

template <typename S>
auto major(
    std::vector<double>& freqs,
    std::vector<HostArray<StokesI<S>, 2>>& residuals,
    const GridSpec imgGridspec,
    std::vector<HostArray<thrust::complex<S>, 2>>& psfs,
    const GridSpec psfGridspec,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter
) {
    switch (freqs.size()) {
    case 1:
        return clean::_major<S, 1>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 2:
        return clean::_major<S, 2>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 3:
        return clean::_major<S, 3>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 4:
        return clean::_major<S, 4>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 5:
        return clean::_major<S, 5>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 6:
        return clean::_major<S, 6>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 7:
        return clean::_major<S, 7>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 8:
        return clean::_major<S, 8>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 9:
        return clean::_major<S, 9>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 10:
        return clean::_major<S, 10>(
            freqs, residuals, imgGridspec, psfs, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    default:
        fmt::println(
            stderr, "Too many channel groups (maximum: 10)"
        );
        abort();
    }
}

}