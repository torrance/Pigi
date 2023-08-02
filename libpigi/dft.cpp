#include "dft.h"

#include <complex>

#include <hip/hip_runtime.h>

#include "outputtypes.h"
#include "util.h"

template <typename T, typename S>
__global__ void _idft(
    DeviceSpan<T, 2> img,
    DeviceSpan<UVDatum<S>, 1> uvdata,
    GridSpec gridspec,
    S normfactor
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
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
    DeviceSpan<T, 2> img,
    DeviceSpan<UVDatum<S>, 1> uvdata,
    GridSpec gridspec,
    S normfactor
) {
    auto fn = _idft<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        img, uvdata, gridspec, normfactor
    );
}

// Explicit template instantiations

template void idft(
    DeviceSpan<StokesI<float>, 2> img,
    DeviceSpan<UVDatum<float>, 1> uvdata,
    GridSpec gridspec,
    float normfactor
);

template void idft(
    DeviceSpan<StokesI<double>, 2> img,
    DeviceSpan<UVDatum<double>, 1> uvdata,
    GridSpec gridspec,
    double normfactor
);

template void idft(
    DeviceSpan<ComplexLinearData<float>, 2> img,
    DeviceSpan<UVDatum<float>, 1> uvdata,
    GridSpec gridspec,
    float normfactor
);

template void idft(
    DeviceSpan<ComplexLinearData<double>, 2> img,
    DeviceSpan<UVDatum<double>, 1> uvdata,
    GridSpec gridspec,
    double normfactor
);