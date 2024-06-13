#pragma once

#include <array>
#include <cmath>
#include <chrono>
#include <optional>
#include <ranges>
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
#include "gslfit.h"
#include "hip.h"
#include "logger.h"
#include "memory.h"
#include "mset.h"
#include "outputtypes.h"

namespace clean {

using LMpx = std::tuple<long long, long long>;

struct XYHash {
    std::size_t operator()(const LMpx& xy) const noexcept {
        auto& [x, y] = xy;
        size_t hash = std::hash<long long>{}(x);
        hash ^= std::hash<long long>{}(y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

/**
 * A utility function for finding which pixel 'belongs' to which field.
 * Currently, we choose a field based on the distance to the center of the field.
 * If 1. the field contains the pixel, and 2. its distance is shorter than other fields
 * then we choose this field.
 */
std::optional<size_t> findnearestfield(const std::vector<GridSpec>& gridspecs, LMpx lmpx) {
    auto [lpx, mpx] = lmpx;
    std::size_t mindist {std::numeric_limits<size_t>::max()};
    std::optional<size_t> nearestfieldid {};

    for (size_t fieldid {}; auto& gridspec : gridspecs) {
        auto idx = gridspec.LMpxToLinear(lpx, mpx);
        if (idx) {
            size_t dist = (
                std::pow(gridspec.deltalpx - lpx, 2) + std::pow(gridspec.deltampx - mpx, 2)
            );
            if (dist < mindist) {
                mindist = dist;
                nearestfieldid = fieldid;
            }
        }
        ++fieldid;
    }

    return nearestfieldid;
}

template <typename T>
using ComponentMap = std::unordered_map<LMpx, T, XYHash>;

template <typename S, int N>
auto _major(
    std::vector<MeasurementSet::FreqRange>& freqs,
    std::vector<std::vector<HostArray<StokesI<S>, 2>>>& residualss,
    const std::vector<GridSpec>& imgGridspecs,
    std::vector<std::vector<HostArray<thrust::complex<S>, 2>>>& psfss,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter,
    const int spectralparams
) {
    // TODO: Enforce some input requirements

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
    S maxVal {};
    {
        // Combine all images into a single vector containing mean value across channels
        // Where fields overlap, we use findnearestfield() to choose.
        std::vector<S> imgCombined;
        imgCombined.reserve(imgGridspecs.at(0).size());

        for (size_t fieldid {}; fieldid < imgGridspecs.size(); ++fieldid) {
            auto imgGridspec = imgGridspecs[fieldid];
            auto& residuals = residualss[fieldid];

            for (size_t i {}; i < imgGridspec.size(); ++i) {
                // In the case of duplicate pixels, choose most central pixel
                auto lmpx = imgGridspec.linearToLMpx(i);
                if (fieldid == findnearestfield(imgGridspecs, lmpx).value()) {
                    S val {};
                    for (size_t n {}; n < N; ++n) val += residuals[n][i].I.real();
                    imgCombined.push_back(val / N);
                }
            }
        }

        // stddev = sqrt(variance) == sqrt(sum over N (val_N - mean)^2 / N)
        // So we need caculate the mean and variance first
        S mean {};
        for (auto& val : imgCombined) { mean += val; }
        mean /= imgCombined.size();

        S variance {};
        for (auto& val : imgCombined) { variance += std::pow(val - mean, 2); }
        variance /= imgCombined.size();

        // Finally we have the noise estimate
        noise = std::sqrt(variance);

        // Find initial maxVal
        maxVal = std::abs(*std::max_element(
            imgCombined.begin(), imgCombined.end(), [] (auto& lhs, auto& rhs) {
                return std::abs(lhs) < std::abs(rhs);
            }
        ));
    }

    // Set threshold as max of the possible methods
    S threshold = (1 - majorgain) * maxVal;
    bool finalMajor {false};

    if (threshold < noise * autoThreshold || threshold < cleanThreshold) {
        threshold = std::max(noise * autoThreshold, cleanThreshold);
        finalMajor = true;
    }

    Logger::info(
        "Beginning{}major clean cycle: from {:.2g} Jy to {:.2g} Jy (est. noise {:.2g} Jy)",
        finalMajor ? " (final) " : " ", maxVal, threshold, noise
    );

    // Reduce search space only to existing pixels above the threshold;
    // Additionally, reorder the pixels so that channel values are stored contiguously
    using ChannelValues = std::array<S, N>;
    std::vector<std::tuple<LMpx, ChannelValues>> pixels;
    for (size_t fieldid {}; fieldid < imgGridspecs.size(); ++fieldid) {
        auto imgGridspec = imgGridspecs[fieldid];

        for (size_t i {}; i < imgGridspec.size(); ++i) {
            auto lmpx = imgGridspec.linearToLMpx(i);
            if (fieldid != findnearestfield(imgGridspecs, lmpx).value()) continue;

            ChannelValues vals {};
            for (size_t n {}; n < N; ++n) {
                vals[n] = residualss[fieldid][n][i].I.real();
            }

            auto meanVal = std::apply([] (auto... vals) {
                return ((vals + ...)) / N;
            }, vals);

            if (std::abs(meanVal) >= 0.9 * threshold) {
                pixels.emplace_back(lmpx, vals);
            }
        }
    }

    // Rearrange PSFs so that channel values are stored contiguously
    std::vector<GridSpec> psfGridspecs(psfss.size());
    std::vector<HostArray<ChannelValues, 2>> psfsDense(psfss.size());

    for (size_t fieldid {}; fieldid < psfss.size(); ++fieldid) {
        psfGridspecs[fieldid] = {
            .Nx=psfss.at(fieldid).at(0).size(0),
            .Ny=psfss.at(fieldid).at(0).size(1)
        };

        HostArray<ChannelValues, 2> psfDense {psfGridspecs[fieldid].shape()};
        for (size_t n {}; n < N; ++n) {
            auto& psf = psfss[fieldid][n];
            for (size_t i {}; i < psf.size(); ++i) {
                psfDense[i][n] = psf[i].real();
            }
        }
        psfsDense[fieldid] = std::move(psfDense);
    }

    // Creating fitting object
    const int nparams = std::min(spectralparams, N); // cap nparams at N channels
    GSLFit fitter([] (const gsl_vector* params, void* data, gsl_vector* residual) -> int {
        using Data = std::tuple<std::vector<MeasurementSet::FreqRange>, ChannelValues>;
        auto& [freqs, vals] = *static_cast<Data*>(data);

        for (size_t n {}; n < N; ++n) {
            auto& [freq_low, freq_high] = freqs[n];
            double model {};
            for (size_t order {1}; order <= params->size; ++order) {
                // This model is a polynomial fit over channel(s) having
                // finite (i.e. not infinitesimal) bandwidth.
                // See e.g. Offringa & Smirnoff 2017, section 2.1.
                model += gsl_vector_get(params, order - 1) * (
                    std::pow(freq_high, order) - std::pow(freq_low, order)
                ) / (order * (freq_high - freq_low));
            }
            gsl_vector_set(residual, n, vals[n] - model);
        }

        return GSL_SUCCESS;
    }, nparams, N);

    // Pre allocate objects used in fitting
    std::vector<double> params0(nparams);
    auto datapair = std::make_tuple(freqs, ChannelValues{});

    size_t iter {};
    for (; iter < niter; ++iter) {
        auto start = std::chrono::steady_clock::now();

        // Find the maximum value
        // We use std::accumulate as a left fold
        auto [maxIdx, maxMeanVal, maxVals] = std::accumulate(
            pixels.begin(), pixels.end(),
            std::make_tuple(LMpx {}, S(0), ChannelValues {}),
            [] (auto&& acc, auto& pixel) {
                auto [lmpx, vals] = pixel;
                auto meanVal = std::apply([] (auto... vals) {
                    return std::abs((vals + ...)) / N;
                }, vals);

                if (meanVal > std::get<1>(acc)) {
                    return std::make_tuple(lmpx, meanVal, vals);
                } else {
                    return acc;
                }
            }
        );

        // Recalculate mean value with correct sign (for logging)
        maxVal = std::apply([] (auto... vals) { return ((vals + ...)) / N; }, maxVals);

        maxFindingDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start
        );

        if (nparams == 1 || N == nparams) {
            // No fit required; just apply gain
            for (auto& val : maxVals) val *= static_cast<S>(minorgain);
        } else {
            // Fit polynomial to output
            params0[0] = maxVal;
            std::get<1>(datapair) = maxVals;
            auto& coeffs = fitter.fit(params0, &datapair);

            // Evalate the model and apply gain
            for (size_t n {}; n < N; ++n) {
                maxVals[n] = 0;
                auto& [freq_low, freq_high] = freqs[n];
                for (size_t order {1}; order <= coeffs.size(); ++order) {
                    // This is the same polynomial model as used in GSLFit() above.
                    maxVals[n] +=  coeffs[order - 1] * (
                        std::pow(freq_high, order) - std::pow(freq_low, order)
                    ) / (order * (freq_high - freq_low));
                }
                maxVals[n] *= static_cast<S>(minorgain);
            }
        }

        start = std::chrono::steady_clock::now();

        // Save component
        for (size_t n {}; n < N; ++n) {
            components[n][maxIdx] += StokesI<S>(maxVals[n]);
        }

        // Separate peak grid location into components
        auto [lpeak, mpeak] = maxIdx;

        // Subtract (component * psf) from pixels
        for (auto& [lmpx, vals] : pixels) {
            auto [xpx, ypx] = lmpx;

            // Find the nearest PSF to pixel
            std::ranges::iota_view<size_t, size_t> fieldids{0, psfGridspecs.size()};
            size_t nearestfieldid = *std::min_element(
                fieldids.begin(), fieldids.end(), [&] (size_t lhs, size_t rhs) {
                    auto& lhsgridspec = psfGridspecs[lhs];
                    auto& rhsgridspec = psfGridspecs[rhs];

                    auto lhsdist = (
                        std::pow(lhsgridspec.deltalpx - xpx, 2) +
                        std::pow(lhsgridspec.deltampx - ypx, 2)
                    );
                    auto rhsdist = (
                        std::pow(rhsgridspec.deltalpx - xpx, 2) +
                        std::pow(rhsgridspec.deltampx - ypx, 2)
                    );

                    return lhsdist < rhsdist;
                }
            );

            auto& psfGridspec = psfGridspecs[nearestfieldid];
            auto& psfDense = psfsDense[nearestfieldid];

            // Calculate respective xpx, ypx in psf image
            // First, find offset from peak center
            xpx -= lpeak;
            ypx -= mpeak;

            // Convert psf coordinates wrt to bottom left corner
            xpx += psfGridspec.Nx / 2;
            ypx += psfGridspec.Ny / 2;

            // Subtract component from pixel values if it maps to a valid psf pixel
            // It is possible no psf maps to a pixel, since psfs have been heavily
            // cropped in size to just their brightest pixels.
            if (0 <= xpx && xpx < psfGridspec.Nx && 0 <= ypx && ypx < psfGridspec.Ny) {
                auto idx = psfGridspec.gridToLinear(xpx, ypx);

                auto psf = psfDense[idx];
                for (size_t n {}; n < N; ++n) {
                    vals[n] -= maxVals[n] * psf[n];
                }
            }
        }

        psfSubtractionDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start
        );

        if (iter == 0) Logger::info(
            "[Initial iteration] {:.3f} Jy peak found; search space {} pixels",
            maxVal, pixels.size()
        );

        if ((iter + 1) % 1000 == 0) {
            // Filter list for any pixels that have now fallen beneath the threshold
            std::erase_if(pixels, [=] (auto& pixel) {
                auto& [_, vals] = pixel;
                auto meanVal = std::apply([] (auto... vals) {
                    return ((vals + ...)) / N;
                }, vals);
                return std::abs(meanVal) < 0.9 * threshold;
            });

            Logger::info(
                "[{} iteration] {:.3f} Jy peak found; search space {} pixels",
                iter + 1, maxVal, pixels.size()
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

    Logger::info(
        "Clean cycle complete ({} iterations this major cycle; {:.2f} s peak finding; {:.2f} s PSF subtraction). Subtracted flux: {:.4g} Jy",
        iter, maxFindingDuration.count() / 1e9,
        psfSubtractionDuration.count() / 1e9, subtractedFlux
    );

    return std::make_tuple(std::move(components), iter, finalMajor);
}

template <typename S>
auto major(
    std::vector<MeasurementSet::FreqRange>& freqs,
    std::vector<std::vector<HostArray<StokesI<S>, 2>>>& residualss,
    const std::vector<GridSpec>& imgGridspecs,
    std::vector<std::vector<HostArray<thrust::complex<S>, 2>>>& psfss,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter,
    const int spectralparams
) {
    switch (freqs.size()) {
    case 1:
        return clean::_major<S, 1>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 2:
        return clean::_major<S, 2>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 3:
        return clean::_major<S, 3>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 4:
        return clean::_major<S, 4>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 5:
        return clean::_major<S, 5>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 6:
        return clean::_major<S, 6>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 7:
        return clean::_major<S, 7>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 8:
        return clean::_major<S, 8>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 9:
        return clean::_major<S, 9>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    case 10:
        return clean::_major<S, 10>(
            freqs, residualss, imgGridspecs, psfss, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter, spectralparams
        );
        break;
    default:
        throw std::runtime_error("Too many channel groups (maximum: 10)");
    }
}

}