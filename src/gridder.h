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
__global__ __launch_bounds__(128)
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
    // To increase the computational intensity of the kerne (just slightly),
    // we compute $nchunk pixels at once.
    const int nchunk {4};

    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<float>) + sizeof(S))
    ];
    auto data_cache = reinterpret_cast<ComplexLinearData<float>*>(_cache);
    auto invlambdas_cache = reinterpret_cast<S*>(data_cache + cachesize);

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

        const size_t rowstride = lambdas.size();

        for (
            size_t idx = blockIdx.x * blockDim.x * nchunk + threadIdx.x * nchunk;
            idx < blockDim.x * nchunk * cld<size_t>(subgridsize, blockDim.x * nchunk);
            idx += blockDim.x * gridDim.x * nchunk
        ) {
            std::array<std::array<S, 3>, nchunk> lmns;
            for (int i {}; i < nchunk; ++i) {
                auto [l, m] = subgridspec.linearToSky<S>(idx + i);
                S n {ndash(l, m)};
                lmns[i] = {l, m, n};
            }

            std::array<ComplexLinearData<S>, nchunk> cells {};

            for (size_t irow {rowstart}; irow < rowend; ++irow) {
                auto uvw = uvws[irow];
                S u = std::get<0>(uvw), v = std::get<1>(uvw), w = std::get<2>(uvw);

                // Precompute theta in _meters_ By doing this here, we can compute the true
                // theta by a multiplication (by inverse lambda) in the host path.
                // thetaoffset, on the other hand, can be fully computed at this point.
                std::array<S, nchunk> thetas;
                std::array<S, nchunk> thetaoffsets;
                for (int i {}; i < nchunk; ++i) {
                    auto [l, m, n] = lmns[i];
                    thetas[i] = {2 * ::pi_v<S> * (u * l + v * m + w * n)};  // [meters]
                    thetaoffsets[i] = {2 * ::pi_v<S> * (u0 * l + v0 * m + w0 * n)};  // [dimensionless]
                }

                // We force all data to have positive w values to reduce the number of w layers,
                // since V(u, v, w) = V(-u, -v, -w)^H
                // The data has already been partitioned making this assumption.
                if (w < 0) for (int i {}; i < nchunk; ++i) {
                    thetas[i] *= -1;
                }

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

                    // Read through cache
                    for (size_t j {}; j < N; ++j) {
                        // Retrieve value of uvdatum from the cache
                        // This shared mem load is broadcast across the warp and so we
                        // don't need to worry about bank conflicts
                        const auto datum = static_cast<ComplexLinearData<S>>(data_cache[j]);
                        const auto invlamda = invlambdas_cache[j];

                        for (int i {}; i < nchunk; ++i) {
                            const auto phase = cis(thetas[i] * invlamda - thetaoffsets[i]);

                            // Equivalent of: cell += uvdata.data * phase
                            // Written out explicitly to use fma operations
                            cmac(cells[i].xx, datum.xx, phase);
                            cmac(cells[i].yx, datum.yx, phase);
                            cmac(cells[i].xy, datum.xy, phase);
                            cmac(cells[i].yy, datum.yy, phase);
                        }
                    }
                }
            }

            for (int i {}; i < nchunk && idx + i < subgridsize; ++i) {
                T output;
                if (makePSF) {
                    // No beam correction for PSF
                    output = static_cast<T>(cells[i]);
                } else {
                    // Grab A terms and apply beam corrections and normalization
                    const auto Al = static_cast<ComplexLinearData<S>>(
                        alefts[wid][idx + i]
                    ).inv();
                    const auto Ar = static_cast<ComplexLinearData<S>>(
                        arights[wid][idx + i]
                    ).inv().adjoint();

                    // Apply beam to cell: inv(Aleft) * cell * inv(Aright)^H
                    // Then conversion from LinearData to output T
                    output = static_cast<T>(
                        matmul(matmul(Al, cells[i]), Ar)
                    );

                    // Calculate norm
                    T norm = matmul(Al, Ar).norm();

                    // Finally, apply norm
                    output /= norm;
                }

                // Perform final FFT fft normalization and apply taper
                output *= subtaper[idx + i] / subgridsize;

                subgrids[wid * subgridsize + idx + i] = output;
            }  // loop: chunks
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