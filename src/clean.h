#pragma once

#include <cmath>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"

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

struct Config {
    double minorgain {0.1};
    double majorgain {0.8};
    double threshold {0};
    size_t niter {std::numeric_limits<size_t>::max()};
};

template <typename S>
std::tuple<HostArray<StokesI<S>, 2>, S, size_t> major(
    HostSpan<StokesI<S>, 2> img,
    const GridSpec imgGridspec,
    const HostSpan<thrust::complex<S>, 2> psf,
    const GridSpec psfGridspec,
    const Config config
) {
    HostArray<StokesI<S>, 2> components {img.shape()};

    // Clean down to either:
    //   1. the explicit threshold limit, or
    //   2. the current peak value minus the majorgain
    // (whichever is greater).
    S maxInit {};
    for (auto& val : img) {
        maxInit = std::max(maxInit, std::abs(val.I.real()));
    }

    auto threshold = std::max((1 - config.majorgain) * maxInit, config.threshold);
    fmt::println(
        "Beginning major clean cycle: from {:.2g} Jy to {:.2g}", maxInit, threshold)
    ;

    // Transfer img and psf to device
    DeviceArray<StokesI<S>, 2> img_d {img};
    DeviceArray<thrust::complex<S>, 2> psf_d {psf};

    size_t iter {};
    while (++iter < config.niter) {
        // Find the device pointer to maximum value
        StokesI<S>* maxptr = thrust::max_element(
            thrust::device, img_d.begin(), img_d.end(), [] __device__ (auto lhs, auto rhs) {
                return abs(lhs.I.real()) < abs(rhs.I.real());
            }
        );
        size_t idx = maxptr - img_d.begin();

        // Copy max value host -> device
        StokesI<S> maxval;
        HIPCHECK(
            hipMemcpyAsync(
                static_cast<void*>(&maxval), static_cast<void*>(maxptr),
                sizeof(StokesI<S>), hipMemcpyDeviceToHost, hipStreamPerThread
            )
        );
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

        // Apply gain
        auto val = maxval.I.real() * static_cast<S>(config.minorgain);
        auto [xpx, ypx] = imgGridspec.linearToGrid(idx);

        // Save component and subtract contribution from image
        components[idx] += val;
        subtractpsf<StokesI<S>, S>(
            img_d, imgGridspec, psf_d, psfGridspec, xpx, ypx, val
        );

        if (iter % 1000 == 0) fmt::println(
            "   [{} iteration] {:.2g} Jy peak found", iter, thrust::abs(maxval.I)
        );

        if (std::abs(maxval.I.real()) <= threshold) break;
    }

    img = img_d;

    maxInit = 0;
    for (auto& val : img) {
        maxInit = std::max(maxInit, std::abs(val.I.real()));
    }
    fmt::println(
        "Clean cycle complete ({} iterations this major cycle). Peak value remaining: {:.2g} Jy",
        iter, maxInit
    );

    return std::make_tuple(std::move(components), maxInit, iter);
}

}