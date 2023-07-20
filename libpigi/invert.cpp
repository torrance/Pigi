#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "array.cpp"
#include "fft.cpp"
#include "gridspec.cpp"
#include "outputtypes.cpp"
#include "uvdatum.cpp"
#include "workunit.cpp"

template <typename T, typename S>
__global__
void _wcorrect(SpanMatrix<T> grid, const GridSpec gridspec, const S w0) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        grid[idx] *= cispi(2 * w0 * ndash(l, m));
    }
}

template <typename T, typename S>
void wcorrect(SpanMatrix<T> grid, const GridSpec gridspec, const S w0) {
    auto fn = _wcorrect<StokesI<S>, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec, w0
    );
}

template<template<typename> typename T, typename S>
HostMatrix<T<S>> invert(
    const SpanVector<WorkUnit<S>> workunits,
    const GridSpec gridspec,
    const SpanMatrix<S> taper,
    const SpanMatrix<S> subtaper
) {
    HostMatrix<T<S>> img {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    HostMatrix<T<S>> wlayer {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};

    DeviceMatrix<T<S>> wlayerd {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    DeviceMatrix<S> subtaperd {subtaper};

    auto plan = fftPlan<T<S>>(gridspec);

    // Get unique w terms
    std::vector<S> ws(workunits.size());
    std::transform(
        workunits.begin(), workunits.end(),
        ws.begin(),
        [](const auto& workunit) { return workunit.w0; }
    );
    std::sort(ws.begin(), ws.end());
    ws.resize(std::unique(ws.begin(), ws.end()) - ws.begin());

    for (const auto w0 : ws) {
        fmt::println("Processing w={} layer...", w0);

        // TOFIX: This makes a copy of workunits, which is fine for now
        // so long as data is a span, but not when it is owned.
        std::vector<WorkUnit<S>> wworkunits;
        std::copy_if(
            workunits.begin(), workunits.end(), std::back_inserter(wworkunits),
            [=](const auto& workunit) { return workunit.w0 == w0; }
        );

        wlayerd.zero();
        gridder<T<S>, S>(wlayerd, wworkunits, subtaperd);

        // FFT the full wlayer
        fftExec(plan, wlayerd, HIPFFT_BACKWARD);

        // Apply w correction
        wcorrect<T<S>, S>(wlayerd, gridspec, w0);

        wlayer = wlayerd;
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        img += wlayer;
    }

    // The final image still has a taper applied. It's time to remove it.
    img /= taper;

    hipfftDestroy(plan);

    return img;
}