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
__global__ __launch_bounds__(128)
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
    // To increase the computational intensity of the kernel (just slightly),
    // we compute $nchunk visibilities at once.
    const int nchunk {4};

    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<S>) + sizeof(std::array<S, 3>) + sizeof(S))
    ];
    auto subgrid_cache = reinterpret_cast<ComplexLinearData<S>*>(_cache);
    auto lmn_cache = reinterpret_cast<std::array<S, 3>*>(subgrid_cache + cachesize);
    auto thetaoffsets_cache = reinterpret_cast<S*>(lmn_cache + cachesize);

    const size_t subgridsize = subgridspec.size();

    // y block index denotes workunit
    for (size_t wid {blockIdx.y}; wid < workunits.size(); wid += gridDim.y) {
        // Get workunit information
        size_t rowstart, rowend, chanstart, chanend;
        S u0, v0, w0;
        {
            const auto workunit = workunits[wid];
            rowstart = workunit.rowstart - rowoffset;
            rowend = workunit.rowend - rowoffset;
            chanstart = workunit.chanstart;
            chanend = workunit.chanend;
            u0 = workunit.u;
            v0 = workunit.v;
            w0 = workunit.w;
        }

        // We process $nchunk visibilities at a time, going along a row.
        // The true channel with of this workunit may not be an even multiple of $nchunk
        // and thus we may calculcate some edge channels that are not part of this
        // workunit. These will be discarded later.
        const size_t nchans = cld<size_t>(chanend - chanstart, nchunk) * nchunk;
        const size_t nvis = (rowend - rowstart) * nchans;

        for (
            size_t idx = blockIdx.x * blockDim.x * nchunk + threadIdx.x * nchunk;
            idx < blockDim.x * nchunk * cld<size_t>(nvis, blockDim.x * nchunk);
            idx += blockDim.x * gridDim.x * nchunk
        ) {
            const size_t irow = idx / nchans + rowstart;
            const size_t ichan = idx % nchans + chanstart;

            // Calculate inverse lambda values for each channel in our chunk
            std::array<S, nchunk> invlambdas;
            for (int i {}; i < nchunk && ichan + i < chanend; ++i) {
                invlambdas[i] = 1 / lambdas[ichan + i];
            }

            // Get u,vw for this datum
            auto [u, v, w] = [&] () -> std::array<S, 3> {
                auto [u, v, w] = uvws[irow];
                return {static_cast<S>(u), static_cast<S>(v), static_cast<S>(w)};
            }();

            // We force all data to have positive w values to reduce the number of w layers,
            // since V(u, v, w) = V(-u, -v, -w)^H
            // The data has already been partitioned making this assumption.
            int signw = w < 0 ? -1 : 1;
            u *= -2 * ::pi_v<S> * signw;
            v *= -2 * ::pi_v<S> * signw;
            w *= -2 * ::pi_v<S> * signw;

            std::array<ComplexLinearData<S>, nchunk> datums {};

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

                    // Precompute theta offset, which we an do since each block shares
                    // the same workunit. [dimensionless]
                    thetaoffsets_cache[j] = -2 * ::pi_v<S> * (u0 * l + v0 * m + w0 * n);
                }
                __syncthreads();

                // Cycle through cache
                for (size_t j {}; j < N; ++j) {
                    // These shared mem loads should be broadcast across the warp and so we
                    // don't need to worry about bank conflicts
                    const auto [l, m, n] = lmn_cache[j];
                    const auto cell = subgrid_cache[j];

                    const S theta {u * l + v * m + w * n};  // [meters]
                    const S thetaoffset {thetaoffsets_cache[j]};  // [dimensionless]

                    for (int k {}; k < nchunk; ++k) {
                        const auto phase = cis(theta * invlambdas[k] - thetaoffset);

                        // Equivalent of: data += cell * phase
                        // Written out explicitly to use fma operations
                        cmac(datums[k].xx, cell.xx, phase);
                        cmac(datums[k].yx, cell.yx, phase);
                        cmac(datums[k].xy, cell.xy, phase);
                        cmac(datums[k].yy, cell.yy, phase);
                    }
                }
            }

            if (irow < rowend) {
                for (int i {}; i < nchunk && ichan + i < chanend; ++i) {
                    // If w was negative, we need to take the adjoint before storing the datum value
                    if (signw == -1) datums[i] = datums[i].adjoint();

                    switch (degridop) {
                    case DegridOp::Add:
                        data[irow * lambdas.size() + ichan + i]
                            += static_cast<ComplexLinearData<float>>(datums[i]);
                        break;
                    case DegridOp::Subtract:
                        data[irow * lambdas.size() + ichan + i]
                            -= static_cast<ComplexLinearData<float>>(datums[i]);
                        break;
                    }
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
    uint32_t nthreadsx {128};
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