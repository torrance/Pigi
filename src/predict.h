#pragma once

#include <set>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "degridder.h"
#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "workunit.h"


template<typename T, typename S>
__global__
void _wdecorrect(DeviceSpan<T, 2> wlayer, const GridSpec gridspec, const S w0) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        wlayer[idx] *= cispi(-2 * w0 * ndash(l, m));
    }
}

template<typename T, typename S>
void wdecorrect(DeviceSpan<T, 2> wlayer, const GridSpec gridspec, const S w0) {
    auto fn = _wdecorrect<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        wlayer, gridspec, w0
    );
}

template <typename T, typename S>
void predict(
    HostSpan<WorkUnit<S>, 1> workunits,
    const HostSpan<T, 2> img,
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper,
    const DegridOp degridop
) {
    // Apply inverse taper
    HostArray<T, 2> imgcopy {img.shape()};
    std::transform(
        img.begin(), img.end(),
        taper.begin(), imgcopy.begin(), [](T val, const S& t) {
            if (t == 0) return T{};
            return val /= t;
    });

    // Transfer subtaper to device
    DeviceArray<S, 2> subtaperd {subtaper};

    // Get unique w terms
    std::set<S> ws;
    for (auto& workunit : workunits) { ws.insert(workunit.w0); }

    auto plan = fftPlan<T>(gridspec);

    for (const S w0 : ws) {
        fmt::println("Processing w={} layer...", w0);

        DeviceArray<T, 2> wlayer {imgcopy};
        wdecorrect<T, S>(wlayer, gridspec, w0);
        fftExec(plan, wlayer, HIPFFT_FORWARD);

        // We use pointers to avoid any kind of copy of the underlying data
        // (since each workunit owns its own data).
        // TODO: use a views filter instead?
        std::vector<WorkUnit<S>*> wworkunits;
        for (auto& workunit : workunits) {
            if (workunit.w0 == w0) wworkunits.push_back(&workunit);
        }

        degridder<T, S>(wworkunits, wlayer, subtaperd, degridop);
    }
}