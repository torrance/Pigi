#pragma once

#include <hip/hip_runtime.h>

#include "datatable.h"
#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"


template <typename T, typename S>
__global__ void _idft(
    DeviceSpan<T, 2> img,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<LinearData<float>, 2> weights,
    const DeviceSpan<ComplexLinearData<S>, 2> jones,
    const GridSpec gridspec,
    const bool normalize
) {
    const size_t cachesize {256};
    __shared__ char _cache[cachesize * (sizeof(ComplexLinearData<float>) + sizeof(S))];
    auto data_cache = reinterpret_cast<ComplexLinearData<float>*>(_cache);
    auto invlambdas_cache = reinterpret_cast<S*>(&data_cache[cachesize]);

    const size_t nrows = uvws.size();
    const size_t nchans = lambdas.size();

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(gridspec.size(), blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        auto n = ndash(l, m);

        ComplexLinearData<S> cell;

        for (size_t irow {}; irow < nrows; ++irow) {
            auto [u, v, w] = uvws[irow];
            S theta = 2 * ::pi_v<S> * (u * l + v * m + w * n);  // [meters]

            for (size_t ichan {}; ichan < nchans; ichan += cachesize) {
                const size_t N = min(cachesize, nchans - ichan);

                // Populate cache
                for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                    data_cache[j] = data[irow * nchans + ichan + j];
                    data_cache[j] *= weights[irow * nchans + ichan + j];
                    invlambdas_cache[j] = 1. / lambdas[ichan + j];
                }
                __syncthreads();

                // Read through cache
                for (size_t j {}; j < N; ++j) {
                    auto datum = static_cast<ComplexLinearData<S>>(data_cache[j]);
                    datum *= cis(theta * invlambdas_cache[j]);
                    cell += datum;
                }
                __syncthreads();
            }
        }

        // Retrieve and apply beam correction
        if (idx < gridspec.size()) {
            auto j = static_cast<ComplexLinearData<S>>(jones[idx]).inv();
            img[idx] = static_cast<T>(
                matmul(matmul(j, cell), j.adjoint())
            );

            if(normalize) {
                T norm = matmul(j, j.adjoint()).norm();
                img[idx] /= norm;
            }
        }
    }
}

template <template <typename> typename T, typename S>
void idft(
    HostSpan<T<S>, 2> img,
    DataTable& tbl,
    const HostSpan<ComplexLinearData<S>, 2> jones,
    const GridSpec gridspec,
    const bool normalize = false
) {
    std::vector<std::array<double, 3>> uvws_h(tbl.nrows());
    for (size_t i {}; auto m : tbl.metadata()) {
        uvws_h[i++] = {m.u, m.v, m.w};
    }

    DeviceArray<T<S>, 2> img_d {img};
    DeviceArray<ComplexLinearData<S>, 2> jones_d {jones};
    DeviceArray<double, 1> lambdas_d(tbl.lambdas());
    DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);
    DeviceArray<ComplexLinearData<float>, 2> data_d(tbl.data());
    DeviceArray<LinearData<float>, 2> weights_d(tbl.weights());

    auto fn = _idft<T<S>, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        img_d, lambdas_d, uvws_d, data_d, weights_d, jones_d, gridspec, normalize
    );

    copy(img, img_d);
}