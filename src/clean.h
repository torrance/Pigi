#pragma once

#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
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

template <typename S, int N>
auto _major(
    std::vector<ChannelGroup<StokesI, S>> channelgroups,
    const GridSpec imgGridspec,
    const GridSpec psfGridspec,
    const double minorgain,
    const double majorgain,
    const double cleanThreshold,
    const double autoThreshold,
    const size_t niter
) {
    // Create components arrays to return
    std::vector<HostArray<StokesI<S>, 2>> components;
    for (size_t i {}; i < channelgroups.size(); ++i) {
        components.push_back(
            HostArray<StokesI<S>, 2> {imgGridspec.Nx, imgGridspec.Ny}
        );
    }

    // Clean down to either:
    //   1. the autothreshold limit, or
    //   1. the explicit threshold limit, or
    //   2. the current peak value minus the majorgain
    // (whichever is greater).

    S threshold {};
    bool finalMajor {};

    {  // New scope: to deallocate combined
        // Combine residuals into sum image
        HostArray<StokesI<S>, 2> combined {imgGridspec.Nx, imgGridspec.Ny};
        for (auto& channelgroup : channelgroups) {
            combined += channelgroup.residual;
        }
        combined /= StokesI<S>(channelgroups.size());

        // Estimate noise
        S mean {};
        for (auto val : combined) {
            mean += val.I.real();
        }
        mean /= combined.size();

        S variance {};
        for (auto val : combined) {
            variance += std::pow(val.I.real() - mean, 2);
        }
        variance /= combined.size();
        S noise = std::sqrt(variance);

        // Find initial maxVal
        S maxVal {};
        for (auto& val : combined) {
            maxVal = std::max(maxVal, std::abs(val.I.real()));
        }

        // Set threshold as max of the possible methods
        threshold = std::max({
            (1 - majorgain) * maxVal, noise * autoThreshold, cleanThreshold
        });

        finalMajor = (1 - majorgain) * maxVal < threshold;

        fmt::println(
            "Beginning{}major clean cycle: from {:.2g} Jy to {:.2g} (est. noise {:.2g} Jy)",
            finalMajor ? " (final) " : " ", maxVal, threshold, noise
        );
    }

    // Transfer residuals and psfs to device
    std::array<DeviceArray<StokesI<S>, 2>, N> residuals;
    std::array<DeviceArray<thrust::complex<S>, 2>, N> psfs;

    for (size_t n {}; n < N; ++n) {
        residuals[n] = DeviceArray<StokesI<S>, 2> {channelgroups[n].residual};
        psfs[n] = DeviceArray<thrust::complex<S>, 2> {channelgroups[n].psf};
    }

    // Create the iterators as thrust::tuple
    const auto residualsBegin = thrust::make_zip_iterator(
        make_tuple_from_array(
            std::apply([] (auto&... residuals) {
                return std::array<StokesI<S>*, N> {
                    residuals.begin()...
                };
            }, residuals)
        )
    );
    const auto residualsEnd = thrust::make_zip_iterator(
        make_tuple_from_array(
            std::apply([] (auto&... residuals) {
                return std::array<StokesI<S>*, N> {
                    residuals.end()...
                };
            }, residuals)
        )
    );

    // Pre-allocate a vector of frequencies and max vals
    std::array<double, N> freqs {};
    for (size_t n {}; n < N; ++n) { freqs[n] = channelgroups[n].midfreq; }
    std::array<S, N> maxVals {};

    size_t iter {1};
    for (; iter < niter; ++iter) {
        // Find the pointer to the maximum value
        auto maxIdxPtr = thrust::max_element(
            thrust::device, residualsBegin, residualsEnd, [=] __device__ (auto lhs, auto rhs) {
                auto fn = [] (auto... xs) {
                    return (xs.I.real() + ... + 0);
                };
                auto lhsSum = apply(fn, lhs);
                auto rhsSum = apply(fn, rhs);
                return std::abs(lhsSum) < std::abs(rhsSum);
            }
        );
        long maxIdx {maxIdxPtr - residualsBegin};

        // Copy max values host -> device
        for (size_t n {}; n < N; ++n) {
            HIPCHECK(
                hipMemcpyAsync(
                    static_cast<void*>(&maxVals[n]), static_cast<void*>(residuals[n].data() + maxIdx),
                    sizeof(StokesI<S>), hipMemcpyDeviceToHost, hipStreamPerThread
                )
            );
        }
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

        // Calculate mean val across each channel
        S meanMaxVal = std::accumulate(
            maxVals.begin(), maxVals.end(), S(0)
        ) / maxVals.size();

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

        // Save component and subtract contribution from image
        for (size_t n {}; n < N; ++n) {
            components[n][maxIdx] += maxVals[n];

            subtractpsf<StokesI<S>, S>(
                residuals[n], imgGridspec, psfs[n], psfGridspec, xpx, ypx, maxVals[n]
            );
        }

        if (iter % 1000 == 0) fmt::println(
            "   [{} iteration] {:.2g} Jy peak found", iter, meanMaxVal
        );

        if (std::abs(meanMaxVal) <= threshold) break;
    }

    // Copy device residuals back to host residuals
    for(size_t n {}; n < N; ++n) {
        channelgroups[n].residual = residuals[n];
    }

    fmt::println(
        "Clean cycle complete ({} iterations this major cycle). Peak value remaining: {:.2g} Jy",
        iter, std::accumulate(maxVals.begin(), maxVals.end(), S(0)) / N
    );

    return std::make_tuple(std::move(components), iter, finalMajor);
}

template <typename S>
auto major(
    std::vector<ChannelGroup<StokesI, S>> channelgroups,
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