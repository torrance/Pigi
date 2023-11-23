#pragma once

#include <hip/hip_runtime.h>

#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"
#include "uvdatum.h"


template <typename T, typename S>
__global__ void _idft(
    DeviceSpan<T, 2> img,
    DeviceSpan<ComplexLinearData<S>, 2> jones,
    DeviceSpan<UVDatum<S>, 1> uvdata,
    GridSpec gridspec
) {
    const size_t cachesize {256};
    __shared__ char _cache[cachesize * sizeof(UVDatum<S>)];
    auto cache = reinterpret_cast<UVDatum<S>*>(_cache);

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(gridspec.size(), blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        auto n = ndash(l, m);

        ComplexLinearData<S> cell;

        for (size_t i {}; i < uvdata.size(); i += cachesize) {
            const size_t N = min(cachesize, uvdata.size() - i);

            // Populate cache
            for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                cache[j] = uvdata[i + j];
            }
            __syncthreads();

            // Read through cache
            for (size_t j {}; j < N; ++j) {
                auto uvdatum = cache[j];

                uvdatum.data *= uvdatum.weights;
                uvdatum.data *= cispi(
                    2 * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
                );
                cell += uvdatum.data;
            }
            __syncthreads();
        }

        // Retrieve and apply beam correction
        if (idx < gridspec.size()) {
            auto j = jones[idx].inv();
            img[idx] = static_cast<T>(
                matmul(matmul(j, cell), j.adjoint())
            );
        }
    }
}

template <template <typename> typename T, typename S>
void idft(
    HostSpan<T<S>, 2> img,
    HostSpan<ComplexLinearData<S>, 2> jones,
    HostSpan<UVDatum<S>, 1> uvdata,
    GridSpec gridspec
) {
    DeviceArray<T<S>, 2> img_d {img};
    DeviceArray<ComplexLinearData<S>, 2> jones_d {jones};
    DeviceArray<UVDatum<S>, 1> uvdata_d {uvdata};

    auto fn = _idft<T<S>, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        img_d, jones_d, uvdata_d, gridspec
    );

    img = img_d;

    // Normalize image based on total weight
    // Accumulation variable requires double precision
    T<double> weightTotal {};
    for (const auto& uvdatum : uvdata) {
        weightTotal += T<double>(uvdatum.weights);
    }
    img /= T<S>(weightTotal);
}