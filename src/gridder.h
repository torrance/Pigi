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

template <typename T, typename S>
__global__
void _gridder(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<LinearData<float>, 2> weights,
    const GridSpec subgridspec,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const size_t rowoffset,
    const bool makePSF
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<float>) + sizeof(S))
    ];
    auto data_cache = reinterpret_cast<ComplexLinearData<float>*>(_cache);
    auto invlambdas_cache = reinterpret_cast<S*>(data_cache + cachesize);

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

        const size_t rowstride = lambdas.size();

        for (
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < blockDim.x * cld<size_t>(subgridspec.size(), blockDim.x);
            idx += blockDim.x * gridDim.x
        ) {
            auto [l, m] = subgridspec.linearToSky<S>(idx);
            S n {ndash(l, m)};

            ComplexLinearData<S> cell {};

            for (size_t irow {rowstart}; irow < rowend; ++irow) {
                auto uvw = uvws[irow];
                S u = std::get<0>(uvw), v = std::get<1>(uvw), w = std::get<2>(uvw);

                // Precompute phase in _meters_
                // We can convert to the dimensionless value later by a single
                // multiplication by the inverse lambda per channel
                S theta {2 * ::pi_v<S> * (u * l + v * m + w * n)};  // [meters]
                S thetaoffset {2 * ::pi_v<S> * (u0 * l + v0 * m + w0 * n)};  // [dimensionless]

                // We force all data to have positive w values to reduce the number of w layers,
                // since V(u, v, w) = V(-u, -v, -w)^H
                // The data has already been partitioned making this assumption.
                if (w < 0) theta *= -1;

                for (size_t ichan {chanstart}; ichan < chanend; ichan += cachesize) {
                    const size_t N = min(cachesize, chanend - ichan);

                    // Populate cache
                    __syncthreads();
                    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                        // Copy global values to shared memory cache
                        data_cache[j] = data[irow * rowstride + ichan + j];
                        invlambdas_cache[j] = 1 / static_cast<S>(lambdas[ichan + j]);

                        auto& datum = data_cache[j];

                        // If making a PSF, replace data with single
                        // point source at image center
                        if (makePSF) {
                            // Predict PSF into projection center
                            S deltal = subgridspec.deltal, deltam = subgridspec.deltam;
                            S deltan = ndash<S>(deltal, deltam);
                            S phase =  -2 * ::pi_v<S> * invlambdas_cache[j] * (
                                u * deltal + v * deltam + w * deltan
                            );
                            auto val = cis(phase);
                            datum = {val, val, val, val};
                        }

                        // Apply weights to data
                        datum *= weights[irow * rowstride + ichan + j];

                        // If we've forced w to be positive, we need to take the adjoint here
                        if (w < 0) datum = datum.adjoint();
                    }
                    __syncthreads();

                    // Zombie threads need to keep filling the cache
                    // but can skip doing any actual work
                    if (idx >= subgridspec.size()) continue;

                    // Read through cache
                    for (size_t j {}; j < N; ++j) {
                        // Retrieve value of uvdatum from the cache
                        // This shared mem load is broadcast across the warp and so we
                        // don't need to worry about bank conflicts
                        auto datum = static_cast<ComplexLinearData<S>>(data_cache[j]);

                        auto phase = cis(theta * invlambdas_cache[j] - thetaoffset);

                        // Equivalent of: cell += uvdata.data * phase
                        // Written out explicitly to use fma operations
                        cmac(cell.xx, datum.xx, phase);
                        cmac(cell.yx, datum.yx, phase);
                        cmac(cell.xy, datum.xy, phase);
                        cmac(cell.yy, datum.yy, phase);
                    }
                }
            }

            // Zombie threads can exit early
            if (idx >= subgridspec.size()) return;

            T output;
            if (makePSF) {
                // No beam correction for PSF
                output = static_cast<T>(cell);
            } else {
                // Grab A terms and apply beam corrections and normalization
                const auto Al = static_cast<ComplexLinearData<S>>(alefts[wid][idx]).inv();
                const auto Ar = static_cast<ComplexLinearData<S>>(arights[wid][idx]).inv().adjoint();

                // Apply beam to cell: inv(Aleft) * cell * inv(Aright)^H
                // Then conversion from LinearData to output T
                output = static_cast<T>(
                    matmul(matmul(Al, cell), Ar)
                );

                // Calculate norm
                T norm = matmul(Al, Ar).norm();

                // Finally, apply norm
                output /= norm;
            }

            // Perform final FFT fft normalization and apply taper
            output *= subtaper[idx] / subgridspec.size();

            subgrids[wid * subgridspec.size() + idx] = output;
        }  // loop: subgrid pixels
    }  // loop: workunits
}

template <typename T, typename S>
void gridder(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<LinearData<float>, 2> weights,
    const GridSpec subgridspec,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const size_t rowoffset,
    const bool makePSF
) {
    auto timer = Timer::get("invert::batch::gridder");

    // x-dimension corresponds to cells in the subgrid
    uint32_t nthreadsx {128}; // hardcoded to match the cache size
    uint32_t nblocksx = cld<size_t>(subgridspec.size(), 4 * nthreadsx);

    // y-dimension corresponds to workunit index
    uint32_t nthreadsy {1};
    uint32_t nblocksy = std::min<size_t>(workunits.size(), 65535);

    auto fn = _gridder<T, S>;
    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy, 1), dim3(nthreadsx, nthreadsy, 1),
        0, hipStreamPerThread,
        subgrids, workunits, uvws, data, weights, subgridspec, lambdas,
        subtaper, alefts, arights, rowoffset, makePSF
    );
}