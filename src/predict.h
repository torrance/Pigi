#pragma once

#include <map>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "degridder.h"
#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "logger.h"
#include "memory.h"
#include "workunit.h"

template <typename T, typename S>
void predict(
    HostSpan<WorkUnit<S>, 1> workunits,
    const HostSpan<T, 2> img,
    const GridConfig gridconf,
    const DegridOp degridop
) {
    auto timer = Timer::get("predict");

    const auto gridspec = gridconf.padded();
    const auto subgridspec = gridconf.subgrid();

    // Copy img (with padding) to device and apply inverse taper
    DeviceArray<T, 2> imgd {resize(img, gridconf.grid(), gridspec)};
    {
        DeviceArray<S, 2> taperd {pswf<S>(gridspec)};
        map([] __device__ (auto& img, const auto t) {
            if (t == 0) img = T{};
            else img /= t;
        }, imgd, taperd);
    }

    // Copy subtaper to device
    DeviceArray<S, 2> subtaperd {pswf<S>(subgridspec)};

    auto plan = fftPlan<T>(gridspec);

    // Create wlayer on device
    DeviceArray<T, 2> wlayer {gridspec.shape(), false};

    // Sort workunits into wlayers
    std::map<S, std::vector<WorkUnit<S>*>> wlayers;
    for (auto& workunit : workunits) {
        wlayers[workunit.w0].push_back(&workunit);
    }

    int nwlayer {};
    for (auto& [w0, wworkunits] : wlayers) {
        auto timer = Timer::get("predict::wlayer");

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

        // Apply w-decorrection and copy to wlayer
        map([w0=w0, gridspec=gridspec] __device__ (auto idx, auto img, auto& wlayer) {
            auto [l, m] = gridspec.linearToSky<S>(idx);
            img *= cispi(-2 * w0 * ndash(l, m));
            wlayer = img;
        }, Iota(), imgd, wlayer);

        fftExec(plan, wlayer, HIPFFT_FORWARD);

        // Reset deltal, deltam shift to visibilities
        map([
            =,
            deltal=static_cast<S>(gridspec.deltal),
            deltam=static_cast<S>(gridspec.deltam)
        ] __device__ (auto idx, auto& wlayer) {
            auto [u, v] = gridspec.linearToUV<S>(idx);
            wlayer *= cispi(-2 * (u * deltal + v * deltam));
        }, Iota(), wlayer);

        degridder<T, S>(wworkunits, wlayer, subtaperd, gridconf, degridop);
    }

    hipfftDestroy(plan);
}