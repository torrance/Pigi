#include "invert.h"

#include <algorithm>
#include <complex>
#include <set>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "fft.h"
#include "gridspec.h"
#include "gridder.cpp"
#include "hip.h"
#include "outputtypes.h"
#include "util.h"
#include "uvdatum.h"

template <typename T, typename S>
__global__
void _wcorrect(DeviceSpan<T, 2> grid, const GridSpec gridspec, const S w0) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = gridspec.linearToSky<S>(idx);
        grid[idx] *= cispi(2 * w0 * ndash(l, m));
    }
}

template <typename T, typename S>
void wcorrect(DeviceSpan<T, 2> grid, const GridSpec gridspec, const S w0) {
    auto fn = _wcorrect<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec, w0
    );
}

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invert(
    const HostSpan<WorkUnit<S>, 1> workunits,
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper
) {
    HostArray<T<S>, 2> img {{gridspec.Nx, gridspec.Ny}};
    HostArray<T<S>, 2> wlayer {{gridspec.Nx, gridspec.Ny}};

    DeviceArray<T<S>, 2> wlayerd {{gridspec.Nx, gridspec.Ny}};
    DeviceArray<S, 2> subtaperd {subtaper};

    auto plan = fftPlan<T<S>>(gridspec);

    // Get unique w terms
    std::set<S> ws;
    for (auto& workunit : workunits) { ws.insert(workunit.w0); }

    for (const S w0 : ws) {
        fmt::println("Processing w={} layer...", w0);

        // We use pointers to avoid any kind of copy of the underlying data
        // (since each workunit owns its own data).
        // TODO: use a views filter instead?
        std::vector<const WorkUnit<S>*> wworkunits;
        for (auto& workunit : workunits) {
            if (workunit.w0 == w0) wworkunits.push_back(&workunit);
        }

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

// Explicit template instantiations

template
HostArray<StokesI<float>, 2> invert(
    const HostSpan<WorkUnit<float>, 1> workunits,
    const GridSpec gridspec,
    const HostSpan<float, 2> taper,
    const HostSpan<float, 2> subtaper
);

template
HostArray<StokesI<double>, 2> invert(
    const HostSpan<WorkUnit<double>, 1> workunits,
    const GridSpec gridspec,
    const HostSpan<double, 2> taper,
    const HostSpan<double, 2> subtaper
);