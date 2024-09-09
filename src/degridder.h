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
    const int nchunk {8};

    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[cachesize * sizeof(ComplexLinearData<S>)];
    auto data_cache = reinterpret_cast<ComplexLinearData<S>*>(_cache);

    const size_t subgridsize = subgridspec.size();
    const size_t rowstride = lambdas.size();

    // Calculate warp properties
    const unsigned int warpid = threadIdx.x / warpSize;
    const unsigned int warprank = threadIdx.x % warpSize;
    const unsigned int warpranknext = (warprank + 1) % warpSize;

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

        size_t nchans = chanend - chanstart;
        size_t nrows = rowend - rowstart;
        size_t nvis = nrows * nchans;

        for (
            size_t idx = blockIdx.x * blockDim.x * nchunk + threadIdx.x * nchunk;
            idx < blockDim.x * nchunk * cld<size_t>(subgridsize, blockDim.x * nchunk);
            idx += blockDim.x * gridDim.x * nchunk
        ) {
            // Load pixel cells
            std::array<ComplexLinearData<S>, nchunk> cells;
            std::array<std::array<S, 3>, nchunk> lmns;
            std::array<S, nchunk> thetaoffsets;

            for (int i {}; i < nchunk && idx + i < subgridsize; ++i) {
                cells[i] = static_cast<ComplexLinearData<S>>(
                    subgrids[subgridsize * wid + idx + i]
                );

                // Grab A terms
                auto al = static_cast<ComplexLinearData<S>>(alefts[wid][idx + i]);
                auto ar = static_cast<ComplexLinearData<S>>(arights[wid][idx + i]).adjoint();

                // Apply Aterms, normalization, and taper
                cells[i] = matmul(matmul(al, cells[i]), ar);
                cells[i] *= subtaper[idx + i] / subgridsize;

                auto [l, m] = subgridspec.linearToSky<S>(idx + i);
                auto n = ndash(l, m);
                lmns[i] = {l, m, n};

                thetaoffsets[i] = -2 * ::pi_v<S> * (u0 * l + v0 * m + w0 * n);
            }

            // Cycle over $warpSize visibilities until exhausted (rounded up to multiple
            // of $warpSize)
            const size_t N = cld<size_t>(nvis, warpSize) * warpSize;
            for (size_t i {warprank}; i < N; i += warpSize) {
                size_t irow = rowstart + i / nchans;
                size_t ichan = chanstart + i % nchans;

                int wsign {};
                S u {}, v {}, w {};
                ComplexLinearData<S> datum {};
                if (i < nvis) {
                    auto [u_, v_, w_] = uvws[irow];
                    u = u_; v = v_; w = w_;

                    // We force all data to have positive w values to reduce the number of w layers,
                    // since V(u, v, w) = V(-u, -v, -w)^H
                    // The data has already been partitioned making this assumption.
                    wsign = w < 0 ? -1 : 1;

                    S lambda = lambdas[ichan];
                    u *= -2 * ::pi_v<S> * wsign / lambda;
                    v *= -2 * ::pi_v<S> * wsign / lambda;
                    w *= -2 * ::pi_v<S> * wsign / lambda;
                }

                // Reduce the data by daisy-chaining the visbilities around
                // the warp group, each reducing its own set of pixels onto the datum
                for (size_t j {}; j < warpSize; ++j) {
                    for (int k {}; k < nchunk; ++k) {
                        auto [l, m, n] = lmns[k];
                        auto phase = cis(u * l + v * m + w * n - thetaoffsets[k]);

                        // Equivalent of: data += cell * phase
                        // Written out explicitly to use fma operations
                        cmac(datum.xx, cells[k].xx, phase);
                        cmac(datum.yx, cells[k].yx, phase);
                        cmac(datum.xy, cells[k].xy, phase);
                        cmac(datum.yy, cells[k].yy, phase);
                    }

                    // Cyclically shuffle the visibility data around the warp
                    assert(__activemask() == 0xffffffff);
                    wsign = __shfl(wsign, warpranknext);
                    u = __shfl(u, warpranknext);
                    v = __shfl(v, warpranknext);
                    w = __shfl(w, warpranknext);

                    auto ptr = reinterpret_cast<S*>(&datum);
                    for (int k {}; k < 4; ++k) {
                        ptr[k] = __shfl(ptr[k], warpranknext);
                    }
                }

                // If we forced w positive, it's time to take the conjugate transpose
                if (wsign == -1) datum = datum.adjoint();

                // Write results out to shared memory...
                __syncthreads();
                data_cache[threadIdx.x] = datum;

                // ...and then reduce across warps
                __syncthreads();
                for (size_t j {warpSize + threadIdx.x}; j < blockDim.x; j += warpSize) {
                    datum += data_cache[j];
                }

                // Warp 0 is responsible for finally writing to global memory
                if (i < nvis && warpid == 0) {
                    switch (degridop) {
                    case DegridOp::Add:
                        atomicAdd(
                            data.data() + irow * rowstride + ichan,
                            static_cast<ComplexLinearData<float>>(datum)
                        );
                        break;
                    case DegridOp::Subtract:
                        atomicSub(
                            data.data() + irow * rowstride + ichan,
                            static_cast<ComplexLinearData<float>>(datum)
                        );
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
    uint32_t nblocksx = cld<size_t>(subgridspec.size(), 8 * nthreadsx);

    // y-dimension breaks the subgrid down into 8 blocks
    uint32_t nthreadsy {1};
    uint32_t nblocksy = std::min<size_t>(workunits.size(), 65535);

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        data, subgrids, workunits, uvws, lambdas, subtaper, alefts, arights, subgridspec,
        rowoffset, degridop
    );
}