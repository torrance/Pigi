#pragma once

#include <map>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "degridder.h"
#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "workunit.h"

template <typename T, typename S>
void predict(
    HostSpan<WorkUnit<S>, 1> workunits,
    const HostSpan<T, 2> img,
    const GridConfig gridconf,
    const DegridOp degridop
) {
    const auto gridspec = gridconf.padded();
    const auto subgridspec = gridconf.subgrid();

    // Copy img (with padding) to device and apply inverse taper
    DeviceArray<T, 2> imgd {resize(img, gridconf.grid(), gridspec)};
    {
        DeviceArray<S, 2> taperd {kaiserbessel<S>(gridspec)};
        map([] __device__ (auto& img, const auto t) {
            if (t == 0) img = T{};
            else img /= t;
        }, imgd, taperd);
    }

    // Copy subtaper to device
    DeviceArray<S, 2> subtaperd {kaiserbessel<S>(subgridspec)};

    auto plan = fftPlan<T>(gridspec);

    // Create wlayer on device
    DeviceArray<T, 2> wlayer {gridspec.shape()};

    // Sort workunits into wlayers
    std::map<S, std::vector<WorkUnit<S>*>> wlayers;
    for (auto& workunit : workunits) {
        wlayers[workunit.w0].push_back(&workunit);
    }

    int nwlayer {};
    for (auto& [w0, wworkunits] : wlayers) {
        fmt::print("\rProcessing {}/{} w-layer...", ++nwlayer, wlayers.size());
        fflush(stdout);

        // Apply w-decorrection and copy to wlayer
        map([w0=w0, gridspec=gridspec] __device__ (auto idx, auto img, auto& wlayer) {
            auto [l, m] = gridspec.linearToSky<S>(idx);
            img *= cispi(-2 * w0 * ndash(l, m));
            wlayer = img;
        }, Iota(), imgd, wlayer);

        fftExec(plan, wlayer, HIPFFT_FORWARD);
        degridder<T, S>(wworkunits, wlayer, subtaperd, gridconf, degridop);
    }

    fmt::println(" Done.");
}