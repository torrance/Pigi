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

#include "channelgroup.h"
#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "polyfit.h"

namespace clean {

template <typename T, typename S>
__global__ void _subtractpsf(
    DeviceSpan<T, 2> img, const GridSpec imgGridspec,
    const DeviceSpan<thrust::complex<S>, 2> psf, const GridSpec psfGridspec,
    long long xpeak, long long ypeak, S f
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < psfGridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [xpx, ypx] = psfGridspec.linearToGrid(idx);

        // Set origin to center of PSF
        xpx -= static_cast<long long>(psfGridspec.Nx) / 2;
        ypx -= static_cast<long long>(psfGridspec.Ny) / 2;

        // Set origin to bottom left corner of img
        xpx += static_cast<long long>(imgGridspec.Nx) / 2;
        ypx += static_cast<long long>(imgGridspec.Ny) / 2;

        long long xoffset { xpeak - static_cast<long long>(imgGridspec.Nx) / 2 };
        long long yoffset { ypeak - static_cast<long long>(imgGridspec.Ny) / 2 };

        // Now shift based on location of peak
        xpx += xoffset;
        ypx += yoffset;

        if (
            0 <= xpx && xpx < static_cast<long long>(imgGridspec.Nx) &&
            0 <= ypx && ypx < static_cast<long long>(imgGridspec.Ny)
        ) {
            auto cell = psf[idx];
            img[imgGridspec.gridToLinear(xpx, ypx)] -= (cell *= f);
        }
    }
}

template <typename T, typename S>
void subtractpsf(
    DeviceArray<T, 2>& img, const GridSpec imgGridspec,
    const DeviceArray<thrust::complex<S>, 2>& psf, const GridSpec psfGridspec,
    long long xpeak, long long ypeak, S f
) {
    auto fn = _subtractpsf<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, psfGridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        static_cast<DeviceSpan<T, 2>>(img), imgGridspec,
        static_cast<DeviceSpan<thrust::complex<S>, 2>>(psf), psfGridspec,
        xpeak, ypeak, f
    );
}


template <typename F, typename T, size_t... I>
__host__ __device__
auto _apply(F f, T tuple, std::index_sequence<I...>) {
    return f(thrust::get<I>(tuple)...);
}

template <typename F, typename... Ts>
__host__ __device__
auto apply(F f, thrust::tuple<Ts...> tuple) {
    return _apply(
        f, tuple, std::make_index_sequence<thrust::tuple_size<thrust::tuple<Ts...>>::value>{}
    );
}

template <typename T, size_t N, size_t... I>
__host__
auto _make_tuple_from_array(const std::array<T, N>& arr, std::index_sequence<I...>) {
    return thrust::make_tuple(arr[I]...);
}

template <typename T, size_t N>
__host__
auto make_tuple_from_array(const std::array<T, N>& arr) {
    return _make_tuple_from_array(
        arr, std::make_index_sequence<N>{}
    );
}

template <template<typename> typename T, typename P, int N>
auto findMaximum(const std::array<DeviceArray<T<P>, 2>, N>& imgs) {
    // Create the iterators as thrust::tuple
    const auto imgsBegin = thrust::make_zip_iterator(
        make_tuple_from_array(
            std::apply([] (auto&... imgs) {
                return std::array<const T<P>*, N> {
                    imgs.begin()...
                };
            }, imgs)
        )
    );
    const auto imgsEnd = thrust::make_zip_iterator(
        make_tuple_from_array(
            std::apply([] (auto&... imgs) {
                return std::array<const T<P>*, N> {
                    imgs.end()...
                };
            }, imgs)
        )
    );

    // Find the pointer to the maximum value
    auto maxIdxPtr = thrust::max_element(
        thrust::device, imgsBegin, imgsEnd, [=] __device__ (auto lhs, auto rhs) {
            auto fn = [] (auto... xs) {
                return (xs.I.real() + ... + 0);
            };
            auto lhsSum = apply(fn, lhs);
            auto rhsSum = apply(fn, rhs);
            return std::abs(lhsSum) < std::abs(rhsSum);
        }
    );
    long maxIdx {maxIdxPtr - imgsBegin};

    // Copy max values host -> device
    std::array<P, N> maxVals;
    for (size_t n {}; n < N; ++n) {
        HIPCHECK(
            hipMemcpyAsync(
                static_cast<void*>(&maxVals[n]), const_cast<void *>(static_cast<const void*>(imgs[n].data() + maxIdx)),
                sizeof(T<P>), hipMemcpyDeviceToHost, hipStreamPerThread
            )
        );
    }
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

    return std::make_tuple(maxIdx, maxVals);
}

template <typename T>
using ComponentMap = std::unordered_map<size_t, T>;

template <typename S, int N>
auto _major(
    std::vector<ChannelGroup<StokesI, S>>& channelgroups,
    const GridSpec imgGridspec,
    const GridSpec psfGridspec,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter
) {
    std::chrono::microseconds maxFindingDuration {};
    std::chrono::microseconds psfSubtractionDuration {};

    // Create components maps to return
    // We use maps here since component maps are usually sparse
    std::vector<ComponentMap<StokesI<S>>> components(N);

    // Clean down to either:
    //   1. the autothreshold limit, or
    //   1. the explicit threshold limit, or
    //   2. the current peak value minus the majorgain
    // (whichever is greater).

    // Estimate noise
    S mean {};
    for (auto& channelgroup : channelgroups) {
        for (auto val : channelgroup.residual) {
            mean += val.I.real();
        }
    }
    mean /= (N * imgGridspec.size());

    S variance {};
    for (auto& channelgroup : channelgroups) {
        for (auto val : channelgroup.residual) {
            variance += std::pow(val.I.real() - mean, 2);
        }
    }
    variance /= (N * imgGridspec.size());
    S noise = std::sqrt(variance);

    // Find initial maxVal
    S maxVal {};
    for (size_t idx {}; idx < imgGridspec.size(); ++idx) {
        S val {};
        for (auto& channelgroup : channelgroups) {
            val += channelgroup.residual[idx].I.real();
        }
        maxVal = std::max(maxVal, std::abs(val / N));
    }

    // Set threshold as max of the possible methods
    S threshold = std::max({
        (1 - majorgain) * maxVal, noise * autoThreshold, cleanThreshold
    });
    bool finalMajor = (1 - majorgain) * maxVal < threshold;

    fmt::println(
        "Beginning{}major clean cycle: from {:.2g} Jy to {:.2g} (est. noise {:.2g} Jy)",
        finalMajor ? " (final) " : " ", maxVal, threshold, noise
    );

    // Transfer residuals and psfs to device
    std::array<DeviceArray<StokesI<S>, 2>, N> residuals;
    std::array<DeviceArray<thrust::complex<S>, 2>, N> psfs;

    for (size_t n {}; n < N; ++n) {
        residuals[n] = DeviceArray<StokesI<S>, 2> {channelgroups[n].residual};
        psfs[n] = DeviceArray<thrust::complex<S>, 2> {channelgroups[n].psf};
    }

    // Pre-allocate a vector of frequencies and max vals
    std::array<S, N> freqs {};
    for (size_t n {}; n < N; ++n) { freqs[n] = channelgroups[n].midfreq; }

    size_t iter {1};
    for (; iter < niter; ++iter) {
        auto start = std::chrono::steady_clock::now();

        auto [maxIdx, maxVals] = findMaximum<StokesI, S, N>(residuals);

        maxFindingDuration += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start
        );

        // Calculate mean val across each channel
        S meanMaxVal = std::accumulate(
            maxVals.begin(), maxVals.end(), S(0)
        ) / N;

        // Fit polynomial to output
        // TODO: make root configurable
        auto coeffs = polyfit<S>(freqs, maxVals, std::min(N, 2));

        // Evalation polynomial model and apply gain
        for (size_t n {}; n < N; ++n) {
            maxVals[n] = 0;
            for (size_t i {}; i < coeffs.size(); ++i) {
                maxVals[n] += static_cast<S>(minorgain) * coeffs[i] * std::pow(freqs[n], i);
            }
        }

        // Find grid locations for max index
        auto [xpx, ypx] = imgGridspec.linearToGrid(maxIdx);

        start = std::chrono::steady_clock::now();

        // Save component and subtract contribution from image
        for (size_t n {}; n < N; ++n) {
            components[n][maxIdx] += maxVals[n];

            subtractpsf<StokesI<S>, S>(
                residuals[n], imgGridspec, psfs[n], psfGridspec, xpx, ypx, maxVals[n]
            );
        }

        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        psfSubtractionDuration += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start
        );

        if (iter % 1000 == 0) fmt::println(
            "   [{} iteration] {:.2g} Jy peak found", iter, meanMaxVal
        );

        if (std::abs(meanMaxVal) <= threshold) break;
    }

    // Copy device residuals back to host residuals
    for (size_t n {}; n < N; ++n) {
        channelgroups[n].residual = residuals[n];
    }

    // Find remaining maxVal
    maxVal = 0;
    for (size_t idx {}; idx < imgGridspec.size(); ++idx) {
        S val {};
        for (auto& channelgroup : channelgroups) {
            val += channelgroup.residual[idx].I.real();
        }
        maxVal = std::max(maxVal, std::abs(val / N));
    }

    fmt::println(
        "Clean cycle complete ({} iterations this major cycle; {:.2f} s peak finding; {:.2f} s PSF subtraction). Peak value remaining: {:.2g} Jy",
        iter, maxFindingDuration.count() / 1e6, psfSubtractionDuration.count() / 1e6, maxVal
    );

    return std::make_tuple(std::move(components), iter, finalMajor);
}

template <typename S>
auto major(
    std::vector<ChannelGroup<StokesI, S>>& channelgroups,
    const GridSpec imgGridspec,
    const GridSpec psfGridspec,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter
) {
    switch (channelgroups.size()) {
    case 1:
        return clean::_major<S, 1>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 2:
        return clean::_major<S, 2>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 3:
        return clean::_major<S, 3>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 4:
        return clean::_major<S, 4>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 5:
        return clean::_major<S, 5>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 6:
        return clean::_major<S, 6>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 7:
        return clean::_major<S, 7>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 8:
        return clean::_major<S, 8>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 9:
        return clean::_major<S, 9>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
            cleanThreshold, autoThreshold, niter
        );
        break;
    case 10:
        return clean::_major<S, 10>(
            channelgroups, imgGridspec, psfGridspec, minorgain, majorgain,
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