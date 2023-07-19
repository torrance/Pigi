#pragma once

#include <complex>

#include <hip/hip_runtime.h>

#include "array.cpp"
#include "gridspec.cpp"
#include "util.cpp"
#include "uvdatum.cpp"

template <typename T, typename S>
__global__ void _idft(
    SpanMatrix<T> img,
    SpanVector<UVDatum<S>> uvdata,
    GridSpec gridspec,
    S normfactor 
) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.Nx * gridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        auto n = ndash(l, m);

        ComplexLinearData<S> cell;
        for (auto uvdatum : uvdata) {
            uvdatum.data *= uvdatum.weights;
            uvdatum.data *= cispi(
                2 * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
            );
            cell += uvdatum.data;
        }

        cell /= normfactor;
        img[idx] = cell;
    }
}

template <typename T, typename S>
void idft(
    SpanMatrix<T> img,
    SpanVector<UVDatum<S>> uvdata,
    GridSpec gridspec,
    S normfactor 
) {
    auto fn = _idft<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.Nx * gridspec.Ny
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        img, uvdata, gridspec, normfactor
    );
}