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
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper,
    const DegridOp degridop
) {
    // Copy img to device apply inverse taper
    DeviceArray<T, 2> imgd {img};
    {
        DeviceArray<S, 2> taperd {taper};
        imgd.mapInto([] (auto& img, auto& t) {
            if (t == 0) return T{};
            return img /= t;
        }, imgd.asSpan(), taperd.asSpan());
    }

    // Copy subtaper to device
    DeviceArray<S, 2> subtaperd {subtaper};

    auto plan = fftPlan<T>(gridspec);

    // Create wlayer on device
    DeviceArray<T, 2> wlayer {img.shape()};

    // Sort workunits into wlayers
    std::map<S, std::vector<WorkUnit<S>*>> wlayers;
    for (auto& workunit : workunits) {
        wlayers[workunit.w0].push_back(&workunit);
    }

    for (auto& [w0, wworkunits] : wlayers) {
        fmt::println("Processing w={} layer...", w0);

        // Apply w-decorrection and copy to wlayer
        wlayer.mapInto([w0=w0, gridspec=gridspec] __device__ (auto idx, auto img) {
            auto [l, m] = gridspec.linearToSky<S>(idx);
            return img *= cispi(-2 * w0 * ndash(l, m));
        }, Iota(), imgd.asSpan());

        fftExec(plan, wlayer, HIPFFT_FORWARD);
        degridder<T, S>(wworkunits, wlayer, subtaperd, degridop);
    }
}