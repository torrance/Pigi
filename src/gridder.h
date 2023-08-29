#pragma once

#include <complex>
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
    const int cachesize {256};

    // Workaround for avoiding initialization of shared variables
    __shared__ char smem[cachesize * sizeof(UVDatum<S>)];
    auto uvdatacache = reinterpret_cast<UVDatum<S>*>(smem);

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld(subgridspec.size(), (size_t) blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = subgridspec.linearToSky<S>(idx);
        S n {ndash(l, m)};

        ComplexLinearData<S> cell {};

        for (size_t i {}; i < uvdata.size(); i += cachesize) {
            // Populate cache
            if (threadIdx.x < cachesize && threadIdx.x + i < uvdata.size()) {
                uvdatacache[threadIdx.x] = uvdata[threadIdx.x + i];
            }
            __syncthreads();

            // Read through cache
            for (size_t j {}; j < min(cachesize, uvdata.size() - i); ++j) {
                // Retrieve value from cache
                auto uvdatum = uvdatacache[j];

                auto phase = cispi(
                    2 * (
                        (uvdatum.u - origin.u0) * l +
                        (uvdatum.v - origin.v0) * m +
                        (uvdatum.w - origin.w0) * n
                    )
                );

                if (makePSF) {
                    uvdatum.data = {1, 0, 0, 1};
                }

                uvdatum.data *= uvdatum.weights;
                uvdatum.data *= phase;
                cell += uvdatum.data;
            }
            __syncthreads();
        }

        if (idx < subgridspec.size()) {
            subgrid[idx] = T::fromBeam(cell, Aleft[idx], Aright[idx]);
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
        int nthreads {256}; // hardcoded to match the cache size
        int nblocks = (int) subgridspec.size() / nthreads + 1;
        hipLaunchKernelGGL(
            fn, nblocks, nthreads, 0, hipStreamPerThread,
            subgrid, Aleft, Aright,
            origin, uvdata, subgridspec
        );
    } else {
        auto fn = _gpudift<T, S, false>;
        int nthreads {256}; // hardcoded to match the cache size
        int nblocks = (int) subgridspec.size() / nthreads + 1;
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
    DeviceSpan<T, 2> grid,
    const DeviceSpan<T, 2> subgrid, const GridSpec subgridspec,
    const long long u0px, const long long v0px
) {
    // Create dummy gridspec to have access to gridToLinear() method
    GridSpec gridspec {grid.size(0), grid.size(1), 0, 0};

    auto fn = _addsubgrid<T>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec, subgrid, subgridspec, u0px, v0px
    );
}

template<typename T, typename S>
void gridder(
    DeviceSpan<T, 2> grid,
    const std::vector<const WorkUnit<S>*> workunits,
    const DeviceSpan<S, 2> subtaper,
    const bool makePSF
) {
    const auto subgridspec = workunits.front()->subgridspec;

    // Transfer Aterms to GPU, since these are often shared
    std::unordered_map<
        const ComplexLinearData<S>*, DeviceArray<ComplexLinearData<S>, 2>
    > Aterms;
    for (const auto& workunit : workunits) {
        Aterms.try_emplace(workunit->Aleft.data(), workunit->Aleft);
        Aterms.try_emplace(workunit->Aright.data(), workunit->Aright);
    }

    using Pair = std::tuple<DeviceArray<T, 2>, const WorkUnit<S>*>;
    Channel<const WorkUnit<S>*> workunitsChannel;
    Channel<Pair> subgridsChannel;

    // Enqueue the work units
    for (const auto workunit : workunits) { workunitsChannel.push(workunit); }
    workunitsChannel.close();

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), std::thread::hardware_concurrency());
        ++i
    ) {
        threads.emplace_back([&] {
            // Make FFT plan for each thread
            auto plan = fftPlan<T>(subgridspec);

            while (auto  maybe = workunitsChannel.pop()) {
                // maybe is a std::optional; let's get the value
                const auto workunit = *maybe;

                const UVWOrigin origin {workunit->u0, workunit->v0, workunit->w0};

                // Transfer data to device and retrieve A terms
                const DeviceArray<UVDatum<S>, 1> uvdata {workunit->data};
                const auto& Aleft = Aterms.at(workunit->Aleft.data());
                const auto& Aright = Aterms.at(workunit->Aright.data());

                // Allocate subgrid
                DeviceArray<T, 2> subgrid {{subgridspec.Nx, subgridspec.Ny}};

                // DFT
                gpudift<T, S>(
                    subgrid, Aleft, Aright, origin, uvdata, subgridspec, makePSF
                );

                // Apply taper and perform FFT normalization
                map([N = subgrid.size()] (auto& cell, const auto t) {
                    cell *= (t / N);
                }, subgrid, subtaper);

                // FFT
                fftExec(plan, subgrid, HIPFFT_FORWARD);

                // Sync the stream before we send the subgrid back to the main thread
                HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

                subgridsChannel.push({std::move(subgrid), workunit});
            }

            HIPFFTCHECK( hipfftDestroy(plan) );
        });
    }

    // Meanwhile, process subgrids and add back to the main grid as the become available
    for (size_t i {}; i < workunits.size(); ++i) {
        const auto [subgrid, workunit] = subgridsChannel.pop().value();
        addsubgrid<T>(grid, subgrid, subgridspec, workunit->u0px, workunit->v0px);
    }

    for (auto& t : threads) { t.join(); }
}