#pragma once

#include <algorithm>
#include <complex>
#include <map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "fft.h"
#include "gridspec.h"
#include "gridder.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invert(
    const HostSpan<WorkUnit<S>, 1> workunits,
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper,
    const bool makePSF = false
) {
    DeviceArray<T<S>, 2> imgd {gridspec.Nx, gridspec.Ny};
    DeviceArray<T<S>, 2> wlayerd {gridspec.Nx, gridspec.Ny};
    DeviceArray<S, 2> subtaperd {subtaper};

    auto plan = fftPlan<T<S>>(gridspec);

    // Sort workunits into wlayers
    std::map<S, std::vector<const WorkUnit<S>*>> wlayers;
    for (auto& workunit : workunits) {
        wlayers[workunit.w0].push_back(&workunit);
    }

    for (const auto& [w0, wworkunits] : wlayers) {
        fmt::println("Processing w={} layer...", w0);

        wlayerd.zero();
        gridder<T<S>, S>(wlayerd, wworkunits, subtaperd, makePSF);

        // FFT the full wlayer
        fftExec(plan, wlayerd, HIPFFT_BACKWARD);

        // Apply wcorrection and append layer onto img
        map([gridspec=gridspec, w0=w0] __device__ (auto idx, auto& imgd, auto wlayerd) {
            auto [l, m] = gridspec.linearToSky<S>(idx);
            wlayerd *= cispi(2 * w0 * ndash(l, m));
            imgd += wlayerd;
        }, Iota(), imgd, wlayerd);
    }

    // Copy img to host
    HostArray<T<S>, 2> img {imgd};

    // The final image still has a taper applied. It's time to remove it.
    img /= taper;

    hipfftDestroy(plan);

    return img;
}