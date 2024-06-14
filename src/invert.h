#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "fft.h"
#include "gridspec.h"
#include "gridder.h"
#include "hip.h"
#include "logger.h"
#include "memory.h"
#include "outputtypes.h"
#include "taper.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invert(
    const HostSpan<WorkUnit<S>, 1> workunits,
    const GridConfig gridconf,
    const bool makePSF = false
) {
    // Pad the main gridspec, and create the subgridspec
    const auto gridspec = gridconf.padded();
    const auto subgridspec = gridconf.subgrid();

    // Create device matrices
    DeviceArray<T<S>, 2> imgd {gridspec.shape()};
    DeviceArray<T<S>, 2> wlayerd {gridspec.shape(), false}; // We zero in main loop
    DeviceArray<S, 2> subtaperd {kaiserbessel<S>(subgridspec)};

    auto plan = fftPlan<T<S>>(gridspec);

    // Sort workunits into wlayers
    std::map<S, std::vector<const WorkUnit<S>*>> wlayers;
    for (auto& workunit : workunits) {
        wlayers[workunit.w0].push_back(&workunit);
    }

    int nwlayer {};
    for (const auto& [w0, wworkunits] : wlayers) {
        Logger::verbose("Processing w={} layer ({}/{})...", w0, ++nwlayer, wlayers.size());

        // Prefetch managed memory to GPU
        {
            // Find the bounds of data in this w-layer
            // If uvdata has been sorted by w value, this will be a contiguous region
            UVDatum<S>* low = wworkunits.front()->data.front();
            UVDatum<S>* high = NULL;
            for (auto workunit : wworkunits) {
                for (auto ptr : workunit->data) {
                    low = std::min(low, ptr);
                    high = std::max(high, ptr);
                }
            }

            // Now prefetch
            HIPCHECK( hipMemPrefetchAsync(
                low, (high - low) * sizeof(UVDatum<S>),
                GPU::getInstance().getID(), hipStreamPerThread
            ) );
        }

        wlayerd.zero();
        gridder<T<S>, S>(wlayerd, wworkunits, subtaperd, gridconf, makePSF);

        // Apply deltal, deltam shift
        map([
            =,
            deltal=static_cast<S>(gridspec.deltal),
            deltam=static_cast<S>(gridspec.deltam)
        ] __device__ (auto idx, auto& wlayerd) {
            auto [u, v] = gridspec.linearToUV<S>(idx);
            wlayerd *= cispi(2 * (u * deltal + v * deltam));
        }, Iota(), wlayerd);

        // FFT the full wlayer
        fftExec(plan, wlayerd, HIPFFT_BACKWARD);

        // Apply wcorrection and append layer onto img
        map([gridspec=gridspec, w0=w0] __device__ (auto idx, auto& imgd, auto wlayerd) {
            auto [l, m] = gridspec.linearToSky<S>(idx);
            wlayerd *= cispi(2 * w0 * ndash(l, m));
            imgd += wlayerd;
        }, Iota(), imgd, wlayerd);
    }

    // Copy img to host
    HostArray<T<S>, 2> img {imgd};

    // The final image still has a taper applied. It's time to remove it.
    img /= kaiserbessel<S>(gridspec);

    hipfftDestroy(plan);

    return resize(img, gridconf.padded(), gridconf.grid());
}