#pragma once

#include "hip/hip_runtime.h"

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "timer.h"
#include "workunit.h"

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void _adder(
    DeviceSpan<T, 2> grid, const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits, const DeviceSpan<T, 3> subgrids,
    const GridSpec gridspec, const GridSpec subgridspec
) {
    size_t widx = widxs[blockIdx.y];
    auto workunit = workunits[widx];
    const long long u0px = workunit.upx;
    const long long v0px = workunit.vpx;

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
            size_t grididx = gridspec.gridToLinear(upx, vpx);
            atomicAdd(
                grid.data() + grididx, subgrids[subgridspec.size() * widx + idx]
            );
        }
    }
}

template <typename T>
void adder(
    DeviceSpan<T, 2> grid, const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits, const DeviceSpan<T, 3> subgrids,
    const GridSpec gridspec, const GridSpec subgridspec
) {
    auto timer = Timer::get("invert::wlayers::adder");

    auto fn = _adder<T>;
    auto [nblocksx, nthreadsx] = getKernelConfig(
        fn, subgridspec.size()
    );

    int nthreadsy {1};
    int nblocksy = widxs.size();

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        grid, widxs, workunits, subgrids, gridspec, subgridspec
    );
}