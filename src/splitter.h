#pragma once

#include "hip/hip_runtime.h"

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "timer.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
__global__
void _splitter(
    DeviceSpan<T<S>, 3> subgrids,
    const DeviceSpan<T<S>, 2> grid,
    const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits,
    const GridSpec gridspec,
    const GridSpec subgridspec
) {
    for (size_t yidx {blockIdx.y}; yidx < widxs.size(); yidx += blockDim.y * gridDim.y) {
        size_t widx = widxs[yidx];
        auto workunit = workunits[widx];
        const long long u0px = workunit.upx;
        const long long v0px = workunit.vpx;

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
                // This assignment performs an implicit conversion
                auto val = grid[gridspec.gridToLinear(upx, vpx)];

                // Remove deltal, deltam shift from visibilities
                auto [u, v] = gridspec.gridToUV<S>(upx, vpx);
                val *= cispi(-2 * (u * gridspec.deltal + v * gridspec.deltam));

                subgrids[subgridspec.size() * widx + idx] = val;
            }
        }
    }
}

template <template <typename> typename T, typename S>
auto splitter(
    DeviceSpan<T<S>, 3> subgrids,
    const DeviceSpan<T<S>, 2> grid,
    const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits,
    const GridSpec gridspec,
    const GridSpec subgridspec
) {
    auto timer = Timer::get("predict::wlayers::splitter");

    auto fn = _splitter<T, S>;
    auto [nblocksx, nthreadsx] = getKernelConfig(
        fn, subgridspec.size()
    );

    uint32_t nthreadsy {1};
    uint32_t nblocksy = std::min<size_t>(widxs.size(), 65535);

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        subgrids, grid, widxs, workunits, gridspec, subgridspec
    );
}