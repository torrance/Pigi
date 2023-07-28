#pragma once

#include <complex>
#include <unordered_map>
#include <thread>
#include <tuple>

#include <hip/hip_runtime.h>

#include "channel.cpp"
#include "gridspec.cpp"
#include "memory.cpp"
#include "outputtypes.cpp"
#include "util.cpp"
#include "uvdatum.cpp"
#include "workunit.cpp"

template <typename T, typename S>
__global__
void _gpudift(
    DeviceSpan<T, 2> subgrid,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aleft,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aright,
    const UVWOrigin<S> origin,
    const DeviceSpan< UVDatum<S>, 1 > uvdata,
    const GridSpec subgridspec
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = subgridspec.linearToSky<S>(idx);
        S n {ndash(l, m)};

        ComplexLinearData<S> cell {};
        for (auto uvdatum : uvdata) {
            auto phase = cispi(
                2 * (
                    (uvdatum.u - origin.u0) * l +
                    (uvdatum.v - origin.v0) * m +
                    (uvdatum.w - origin.w0) * n
                )
            );

            uvdatum.data *= uvdatum.weights;
            uvdatum.data *= phase;
            cell += uvdatum.data;
        }

        subgrid[idx] = T::fromBeam(cell, Aleft[idx], Aright[idx]);
    }
}

template <typename T, typename S>
void gpudift(
    DeviceSpan<T, 2> subgrid,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aleft,
    const DeviceSpan< ComplexLinearData<S>, 2 > Aright,
    const UVWOrigin<S> origin,
    const DeviceSpan< UVDatum<S>, 1 > uvdata,
    const GridSpec subgridspec
) {
    auto fn = _gpudift<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        subgrid, Aleft, Aright,
        origin, uvdata, subgridspec
    );
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
    const HostSpan<WorkUnit<S>, 1> workunits,
    const DeviceSpan<S, 2> subtaper
) {
    const auto subgridspec = workunits.front().subgridspec;

    // Transfer Aterms to GPU, since these are often shared
    std::unordered_map<
        const ComplexLinearData<S>*, DeviceArray<ComplexLinearData<S>, 2>
    > Aterms;
    for (const auto& workunit : workunits) {
        Aterms.try_emplace(workunit.Aleft.data(), workunit.Aleft);
        Aterms.try_emplace(workunit.Aright.data(), workunit.Aright);
    }

    using Pair = std::tuple<DeviceArray<T, 2>, const WorkUnit<S>*>;
    Channel<const WorkUnit<S>*> workunitsChannel;
    Channel<Pair> subgridsChannel;

    // Enqueue the work units
    for (const auto& workunit : workunits) { workunitsChannel.push(&workunit); }
    workunitsChannel.close();

    // Ensure all stream operators are complete before spanwing new streams
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), std::thread::hardware_concurrency());
        ++i
    ) {
        threads.emplace_back([&] {
            // Make FFT plan
            auto plan = fftPlan<T>(subgridspec);

            while (auto maybe = workunitsChannel.pop())
            {
                // Get next workunit
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
                    subgrid, Aleft, Aright, origin, uvdata, subgridspec
                );

                // Apply taper and perform FFT normalization
                subgrid.mapInto([N = subgrid.size()] (auto cell, auto t) {
                    return cell *= (t / N);
                }, subgrid.asSpan(), subtaper);

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

        // I don't think this sync should be necessary, since even if grid goes out of scope
        // the call to free() should be enqueued in the stream. In any case, it seems to fix
        // failures.
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
    }

    for (auto& t : threads) { t.join(); }
}