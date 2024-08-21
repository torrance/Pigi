#pragma once

#include <array>

#include "hip/hip_runtime.h"

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "timer.h"
#include "util.h"
#include "workunit.h"

enum class DegridOp {Subtract, Add};

template <typename T, typename S>
__global__
void _degridder(
    DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const GridSpec subgridspec,
    const size_t rowoffset,
    const DegridOp degridop
) {
    // Set up the shared mem cache
    const size_t cachesize {256};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<S>) + sizeof(std::array<S, 3>))
    ];
    auto subgrid_cache = reinterpret_cast<ComplexLinearData<S>*>(_cache);
    auto lmn_cache = reinterpret_cast<std::array<S, 3>*>(subgrid_cache + cachesize);

    const size_t subgridsize = subgridspec.size();

    // y block index denotes workunit
    for (size_t wid {blockIdx.y}; wid < workunits.size(); wid += gridDim.y) {
        // Get workunit information
        size_t rowstart, rowend, chanstart, chanend;
        S u0, v0, w0;
        {
            const auto& workunit = workunits[wid];
            rowstart = workunit.rowstart - rowoffset;
            rowend = workunit.rowend - rowoffset;
            chanstart = workunit.chanstart;
            chanend = workunit.chanend;
            u0 = workunit.u;
            v0 = workunit.v;
            w0 = workunit.w;
        }

        const size_t nchans = chanend - chanstart;
        const size_t nvis = (rowend - rowstart) * nchans;
        const size_t rowstride = lambdas.size();

        for (
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < blockDim.x * cld<size_t>(nvis, blockDim.x);
            idx += blockDim.x * gridDim.x
        ) {
            size_t irow = idx / nchans + rowstart;
            size_t ichan = idx % nchans + chanstart;

            // Get metadata for this datum
            S lambda {};
            std::array<double, 3> uvw;
            if (idx < nvis) {
                lambda = lambdas[ichan];
                uvw = uvws[irow];
            }

            // We force all data to have positive w values to reduce the number of w layers,
            // since V(u, v, w) = V(-u, -v, -w)^H
            // The data has already been partitioned making this assumption.
            int signw = std::get<2>(uvw) < 0 ? -1 : 1;

            // Precompute uvw offsets and convert to wavenumbers
            S u = -2 * ::pi_v<S> * (signw * std::get<0>(uvw) / lambda - u0);
            S v = -2 * ::pi_v<S> * (signw * std::get<1>(uvw) / lambda - v0);
            S w = -2 * ::pi_v<S> * (signw * std::get<2>(uvw) / lambda - w0);

            ComplexLinearData<S> datum;

            for (size_t ipx {}; ipx < subgridsize; ipx += cachesize) {
                const size_t N = min(cachesize, subgridsize - ipx);

                // Populate cache
                __syncthreads();
                for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                    // Load subgrid value; convert to instrumental values
                    auto cell = static_cast<ComplexLinearData<S>>(
                        subgrids[subgridsize * wid + ipx + j]
                    );

                    // Grab A terms
                    auto al = static_cast<ComplexLinearData<S>>(alefts[wid][ipx + j]);
                    auto ar = static_cast<ComplexLinearData<S>>(arights[wid][ipx + j]).adjoint();

                    // Apply Aterms, normalization, and taper
                    cell = matmul(matmul(al, cell), ar);
                    cell *= subtaper[ipx + j] / subgridsize;

                    subgrid_cache[j] = cell;

                    // Precompute l, m, n and cache values
                    auto [l, m] = subgridspec.linearToSky<S>(ipx + j);
                    auto n = ndash(l, m);

                    lmn_cache[j] = {l, m, n};
                }
                __syncthreads();

                // Zombie threads need to keep filling the cache
                // but can skip doing any actual work
                if (idx >= nvis) continue;

                // Cycle through cache
                for (size_t j {}; j < N; ++j) {
                    auto [l, m, n] = lmn_cache[j];
                    auto phase = cis(u * l + v * m + w * n);

                    // Load subgrid cell from the cache
                    // This shared mem load is broadcast across the warp and so we
                    // don't need to worry about bank conflicts
                    auto cell = subgrid_cache[j];

                    // Equivalent of: data += cell * phase
                    // Written out explicitly to use fma operations
                    cmac(datum.xx, cell.xx, phase);
                    cmac(datum.yx, cell.yx, phase);
                    cmac(datum.xy, cell.xy, phase);
                    cmac(datum.yy, cell.yy, phase);
                }
            }

            // If w was negative, we need to take the adjoint before storing the datum value
            if (signw == -1) datum = datum.adjoint();

            if (idx < nvis) {
                switch (degridop) {
                case DegridOp::Add:
                    data[irow * rowstride + ichan]
                        += static_cast<ComplexLinearData<float>>(datum);
                    break;
                case DegridOp::Subtract:
                    data[irow * rowstride + ichan]
                        -= static_cast<ComplexLinearData<float>>(datum);
                    break;
                }
            }
        }  // idx loop
    }  // workunit loop
}

template <typename T, typename S>
void degridder(
    DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const GridSpec subgridspec,
    const size_t rowoffset,
    const DegridOp degridop
) {
    auto timer = Timer::get("predict::batch::degridder");

    auto fn = _degridder<T, S>;

    // x-dimension distributes uvdata
    uint32_t nthreadsx {256};
    uint32_t nblocksx {1};

    // y-dimension breaks the subgrid down into 8 blocks
    uint32_t nthreadsy {1};
    uint32_t nblocksy = std::min<size_t>(workunits.size(), 65535);

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        data, subgrids, workunits, uvws, lambdas, subtaper, alefts, arights, subgridspec,
        rowoffset, degridop
    );
}