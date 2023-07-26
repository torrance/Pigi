#pragma once

#include <algorithm>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "degridder.cpp"
#include "fft.cpp"
#include "gridspec.cpp"
#include "workunit.cpp"


template<typename T, typename S>
__global__
void _wdecorrect(SpanMatrix<T> wlayer, const GridSpec gridspec, const S w0) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        wlayer[idx] *= cispi(-2 * w0 * ndash(l, m));
    }
}

template<typename T, typename S>
void wdecorrect(SpanMatrix<T> wlayer, const GridSpec gridspec, const S w0) {
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
    SpanVector<WorkUnit<S>> workunits,
    const SpanMatrix<T> img,
    const GridSpec gridspec,
    const SpanMatrix<S> taper,
    const SpanMatrix<S> subtaper,
    const DegridOp degridop=DegridOp::Replace
) {
    // Apply inverse taper
    HostMatrix<T> imgcopy {img.shape()};
    std::transform(
        img.begin(), img.end(),
        taper.begin(), imgcopy.begin(), [](T val, const S& t) {
            if (t == 0) return T{};
            return val /= t;
    });

    // Transfer subtaper to device
    DeviceMatrix<S> subtaperd {subtaper};

    // Get unique w terms
    std::vector<S> ws(workunits.size());
    std::transform(
        workunits.begin(), workunits.end(),
        ws.begin(),
        [](const auto& workunit) { return workunit.w0; }
    );
    std::sort(ws.begin(), ws.end());
    ws.resize(std::unique(ws.begin(), ws.end()) - ws.begin());

    auto plan = fftPlan<T>(gridspec);

    for (const S w0 : ws) {
        fmt::println("Processing w={} layer...", w0);

        DeviceMatrix<T> wlayer {(SpanMatrix<T>) imgcopy};
        wdecorrect<T, S>(wlayer, gridspec, w0);
        fftExec(plan, wlayer, HIPFFT_FORWARD);

        // TOFIX: This makes a copy of workunits, which is fine for now
        // so long as data is a span, but not when it is owned.
        std::vector<WorkUnit<S>> wworkunits;
        std::copy_if(
            workunits.begin(), workunits.end(), std::back_inserter(wworkunits),
            [=](const auto& workunit) { return workunit.w0 == w0; }
        );

        degridder<T, S>(wworkunits, wlayer, subtaperd, degridop);
    }
}