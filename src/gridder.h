#pragma once

#include <unordered_map>
#include <thread>
#include <tuple>

#include <hip/hip_runtime.h>

#include "channel.h"
#include "gridspec.h"
#include "fft.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"

template <typename T, typename S, bool makePSF>
__global__
void _gpudift(
    DeviceSpan<T, 2> subgrid,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aleft,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aright,
    const UVWOrigin<S> origin,
    const DeviceSpan< UVDatum<S>, 1 > uvdata,
    const GridSpec subgridspec
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[cachesize * sizeof(UVDatum<S>)];
    auto cache = reinterpret_cast<UVDatum<S>*>(_cache);

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(subgridspec.size(), blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = subgridspec.linearToSky<S>(idx);
        S n {ndash(l, m)};

        ComplexLinearData<S> cell {};

        for (size_t i {}; i < uvdata.size(); i += cachesize) {
            const size_t N = min(cachesize, uvdata.size() - i);

            // Populate cache
            for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                // Copy global value to shared memory cache
                cache[j] = uvdata[i + j];

                // Precompute some values that will be used by all threads
                UVDatum<S>& uvdatum = cache[j];

                // If making a PSF, replace data with single
                // point source at image center
                if constexpr(makePSF) {
                    // Predict PSF into projection center
                    S deltal = subgridspec.deltal, deltam = subgridspec.deltam;
                    S deltan = ndash<S>(deltal, deltam);
                    auto val = cispi(-2 * (
                        uvdatum.u * deltal + uvdatum.v * deltam + uvdatum.w * deltan
                    ));
                    uvdatum.data = {val, val, val, val};
                }

                // Offset u, v, w
                uvdatum.u -= origin.u0;
                uvdatum.v -= origin.v0;
                uvdatum.w -= origin.w0;

                // Apply weights to data
                uvdatum.data *= uvdatum.weights;
            }
            __syncthreads();

            // Read through cache
            for (size_t j {}; j < N; ++j) {
                // Retrieve value of uvdatum from the cache
                // This shared mem load is broadcast across the warp and so we
                // don't need to worry about bank conflicts
                UVDatum<S> uvdatum = cache[j];

                auto phase = cispi(
                    2 * (uvdatum.u * l + uvdatum.v * m + uvdatum.w * n)
                );

                uvdatum.data *= phase;
                cell += uvdatum.data;
            }
            __syncthreads();
        }

        T output;
        if constexpr(makePSF) {
            // No beam correction for PSF
            output = static_cast<T>(cell);
        } else {
            // Grab A terms and apply beam corrections and normalization
            const auto Al = Aleft[idx].inv();
            const auto Ar = Aright[idx].inv().adjoint();

            // Apply beam to cell: inv(Aleft) * cell * inv(Aright)^H
            // Then conversion from LinearData to output T
            output = static_cast<T>(matmul(matmul(Al, cell), Ar));

            // Calculate norm
            T norm = T(matmul(Al, Ar).norm());

            // Finally, apply norm
            output /= norm;
        }

        if (idx < subgridspec.size()) {
            subgrid[idx] = output;
        }
    }
}

template <typename T, typename S>
void gpudift(
    DeviceSpan<T, 2> subgrid,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aleft,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aright,
    const UVWOrigin<S> origin,
    const DeviceSpan< UVDatum<S>, 1 > uvdata,
    const GridSpec subgridspec,
    const bool makePSF
) {
    if (makePSF) {
        auto fn = _gpudift<T, S, true>;
        int nthreads {128}; // hardcoded to match the cache size
        int nblocks = cld<size_t>(subgridspec.size(), nthreads);
        hipLaunchKernelGGL(
            fn, nblocks, nthreads, 0, hipStreamPerThread,
            subgrid, Aleft, Aright,
            origin, uvdata, subgridspec
        );
    } else {
        auto fn = _gpudift<T, S, false>;
        int nthreads {128}; // hardcoded to match the cache size
        int nblocks = cld<size_t>(subgridspec.size(), nthreads);
        hipLaunchKernelGGL(
            fn, nblocks, nthreads, 0, hipStreamPerThread,
            subgrid, Aleft, Aright,
            origin, uvdata, subgridspec
        );
    }
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void _addsubgrid(
    DeviceSpan<T, 2> grid, const GridSpec gridspec,
    const DeviceSpan<T, 2> subgrid, const GridSpec subgridspec,
    const long long u0px, const long long v0px
) {
    // Iterate over each element of the subgrid
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [upx, vpx] = subgridspec.linearToGrid(idx);

        // Transform to pixel position wrt to master grid
        upx += u0px - subgridspec.Nx / 2;
        vpx += v0px - subgridspec.Ny / 2;

        if (
            0 <= upx && upx < static_cast<long long>(gridspec.Nx) &&
            0 <= vpx && vpx < static_cast<long long>(gridspec.Ny)
        ) {
            grid[gridspec.gridToLinear(upx, vpx)] += subgrid[idx];
        }
    }
}

template <typename T>
void addsubgrid(
    DeviceSpan<T, 2> grid, const DeviceSpan<T, 2> subgrid,
    const GridSpec gridspec, const GridSpec subgridspec,
    const long long u0px, const long long v0px, hipStream_t stream = hipStreamPerThread
) {
    auto fn = _addsubgrid<T>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, stream,
        grid, gridspec, subgrid, subgridspec, u0px, v0px
    );
}

template<typename T, typename S>
void gridder(
    DeviceSpan<T, 2> grid,
    const std::vector<const WorkUnit<S>*> workunits,
    const DeviceSpan<S, 2> subtaper,
    const GridConfig gridconf,
    const bool makePSF
) {
    const GridSpec gridspec = gridconf.padded();
    const GridSpec subgridspec = gridconf.subgrid();

    // Transfer _unique_ Aterms to GPU
    std::unordered_map<
        std::shared_ptr<HostArray<ComplexLinearData<S>, 2>>,
        DeviceArray<ComplexLinearData<S>, 2>
    > Aterms;
    for (const auto& workunit : workunits) {
        Aterms.try_emplace(workunit->Aleft, *workunit->Aleft);
        Aterms.try_emplace(workunit->Aright, *workunit->Aright);
    }

    using Pair = std::tuple<DeviceArray<T, 2>, const WorkUnit<S>*>;
    Channel<const WorkUnit<S>*> workunitsChannel;

    // Enqueue the work units, and note the largest allocation
    size_t largestN {};
    for (const auto workunit : workunits) {
        largestN = std::max(largestN, workunit->data.size());
        workunitsChannel.push(workunit);
    }
    workunitsChannel.close();

    hipStream_t baseStream;
    HIPCHECK( hipStreamCreate(&baseStream) );

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), 3);
        ++i
    ) {
        threads.emplace_back([&] {
            GPU::getInstance().resetDevice(); // needs to be reset for each new thread

            // Create an event object to record when the addsubgrid() kernel has
            // finished, to avoid overwriting the subgrid array before it is fully
            // read.
            hipEvent_t addsubgridDone;
            HIPCHECK( hipEventCreateWithFlags(&addsubgridDone, hipEventBlockingSync) );

            // Make FFT plan for each thread
            auto plan = fftPlan<T>(subgridspec);

            // Allocate device data capable of storing up to the largest workunit
            DeviceArray<UVDatum<S>, 1> uvdata_d(largestN);

            // Allocate subgrid
            DeviceArray<T, 2> subgrid {subgridspec.Nx, subgridspec.Ny};

            while (auto  maybe = workunitsChannel.pop()) {
                // maybe is a std::optional; let's get the value
                const auto workunit = *maybe;

                const UVWOrigin origin {workunit->u0, workunit->v0, workunit->w0};

                // Create span backed by uvdata_d
                DeviceSpan<UVDatum<S>, 1> uvdata_s{
                    {static_cast<long long>(workunit->data.size())}, uvdata_d.pointer()
                };

                // Transfer uvdata host -> device using optimal strategy
                if (workunit->iscontiguous()) {
                    // If uvdata is sorted, we can avoid a bunch of pointer lookups,
                    // and perform a memcopy on the contiguous memory segment.
                    HostSpan<UVDatum<S>, 1> uvdata_h(
                        {static_cast<long long>(workunit->data.size())},
                        workunit->data.front()
                    );
                    copy(uvdata_s, uvdata_h);
                } else {
                    // Otherwise assemble uvdata from (out of order) pointers
                    // and transfer to host
                    HostArray<UVDatum<S>, 1> uvdata_h(workunit->data.size());
                    for (size_t i {}; const auto uvdatumptr : workunit->data) {
                        uvdata_h[i++] = *uvdatumptr;
                    }
                    copy(uvdata_s, uvdata_h);
                }

                // Retrieve A terms that have already been sent to device
                const auto& Aleft = Aterms.at(workunit->Aleft);
                const auto& Aright = Aterms.at(workunit->Aright);

                // Perform DFT, but first ensure addsubgrid() has completed
                HIPCHECK( hipEventSynchronize(addsubgridDone) );
                gpudift<T, S>(
                    subgrid, Aleft, Aright, origin, uvdata_s, subgridspec, makePSF
                );

                // Apply taper and perform FFT normalization
                map([N = subgrid.size()] __device__ (auto& cell, const auto t) {
                    cell *= (t / N);
                }, subgrid, subtaper);

                // FFT
                fftExec(plan, subgrid, HIPFFT_FORWARD);

                // Reset deltal, deltam shift prior to adding to master grid
                map([
                    =,
                    deltal=static_cast<S>(subgridspec.deltal),
                    deltam=static_cast<S>(subgridspec.deltam)
                ] __device__ (auto idx, auto& subgrid) {
                    auto [u, v] = subgridspec.linearToUV<S>(idx);
                    subgrid *= cispi(-2 * (u * deltal + v * deltam));
                }, Iota(), subgrid);

                // Sync the stream before we send the subgrid to addsubgrid()
                // on the steam baseStream
                HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

                addsubgrid<T>(
                    grid, subgrid, gridspec, subgridspec,
                    workunit->u0px, workunit->v0px,
                    baseStream
                );
                HIPCHECK( hipEventRecord(addsubgridDone, baseStream) );
            }

            HIPFFTCHECK( hipfftDestroy(plan) );
            HIPCHECK( hipEventDestroy(addsubgridDone) );
        });
    }

    for (auto& t : threads) { t.join(); }
    HIPCHECK( hipStreamDestroy(baseStream) );
}