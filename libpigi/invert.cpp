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
void wcorrect(T* grid, GridSpec gridspec, S w0) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.Nx * gridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        auto [lpx, mpx] = linearToXY(idx, gridspec);

        auto l {static_cast<S>((lpx - gridspec.Nx / 2) * gridspec.scalelm)};
        auto m {static_cast<S>((mpx - gridspec.Ny / 2) * gridspec.scalelm)};

        grid[idx] *= cispi(2 * w0 * ndash(l, m));
    }
}

template<template<typename> typename T, typename S>
Matrix<T<S>> invert(
    const SpanVector<WorkUnit<S>> workunits,
    const GridSpec gridspec,
    const SpanMatrix<S> taper,
    const SpanMatrix<S> subtaper
) {
    Matrix<T<S>> img {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    Matrix<T<S>> wlayer {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};

    GPUMatrix<T<S>> wlayerd {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    GPUMatrix<S> subtaperd {subtaper};

    // Construct FFT plan
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, T<S>::size(), 1,
        rank, T<S>::size(), 1,
        fftType(wlayerd.data()), T<S>::size()
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

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
        fftshift(wlayerd.data(), gridspec);
        fftExec(plan, wlayerd.data(), HIPFFT_BACKWARD);
        fftshift(wlayerd.data(), gridspec);

        // Apply w correction
        {
            auto fn = wcorrect<StokesI<S>, S>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, gridspec.Nx, gridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, hipStreamPerThread,
                wlayerd.data(), gridspec, w0
            );
        }

        wlayer = wlayerd;
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        img += wlayer;
    }

    // The final image still has a taper applied. It's time to remove it.
    img /= taper;

    hipfftDestroy(plan);

    return img;
}