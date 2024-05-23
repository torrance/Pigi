#pragma once

#include <algorithm>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "channel.h"
#include "fft.h"
#include "hip.h"
#include "memory.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"

enum class DegridOp {Subtract, Add};

template <typename T>
__global__
void _gpudft(
    DeviceSpan<UVDatum<T>, 1> uvdata,
    const UVWOrigin<T> origin,
    const DeviceSpan<ComplexLinearData<T>, 2> subgrid,
    const GridSpec subgridspec,
    const DegridOp degridop
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * sizeof(ComplexLinearData<T>) +
        cachesize * sizeof(std::array<T, 3>)
    ];
    auto cache = reinterpret_cast<ComplexLinearData<T>*>(_cache);
    auto lmns = reinterpret_cast<std::array<T, 3>*>(cache + cachesize);

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(uvdata.size(), blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        UVDatum<T> uvdatum;
        if (idx < uvdata.size()) uvdatum = uvdata[idx];

        // Precompute uvw offsets
        T u = uvdatum.u - origin.u0;
        T v = uvdatum.v - origin.v0;
        T w = uvdatum.w - origin.w0;

        ComplexLinearData<T> data;

        for (size_t i {blockIdx.y * cachesize}; i < subgridspec.size(); i += (gridDim.y * cachesize)) {
            const size_t N = min(cachesize, subgridspec.size() - i);

            // Populate cache
            for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                // Load subgrid value
                cache[j] = subgrid[i + j];

                // Precompute l, m, n and cache values
                auto [l, m] = subgridspec.linearToSky<T>(i + j);
                auto n = ndash(l, m);
                lmns[j] = {l, m, n};
            }
            __syncthreads();

            // Cycle through cache
            for (size_t j {}; j < N; ++j) {
                auto [l, m, n] = lmns[j];
                auto phase = cispi(-2 * (
                    u * l + v * m + w * n
                ));

                // Load subgrid cell from the cache
                // This shared mem load is broadcast across the warp and so we
                // don't need to worry about bank conflicts
                auto cell = cache[j];
                cell *= phase;
                data += cell;
            }

            __syncthreads();
        }

        if (idx < uvdata.size()) {
            switch (degridop) {
            case DegridOp::Add:
                atomicAdd(&uvdata[idx].data, data);
                break;
            case DegridOp::Subtract:
                atomicSub(&uvdata[idx].data, data);
                break;
            }
        }
    }
}

template <typename T>
void gpudft(
    DeviceSpan<UVDatum<T>, 1> uvdata,
    const UVWOrigin<T> origin,
    const DeviceSpan<ComplexLinearData<T>, 2> subgrid,
    const GridSpec subgridspec,
    const DegridOp degridop
) {
    auto fn = _gpudft<T>;

    // x-dimension distributes uvdata
    int nthreadsx {128};
    int nblocksx = cld<size_t>(uvdata.size(), nthreadsx);

    // y-dimension breaks the subgrid down into 8 blocks
    int nthreadsy {1};
    int nblocksy {8};

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        uvdata, origin, subgrid, subgridspec, degridop
    );
}

template <typename T, typename S>
__global__
void _extractSubgrid(
    DeviceSpan<ComplexLinearData<S>, 2> subgrid, const GridSpec subgridspec,
    const DeviceSpan<T, 2> grid, const GridSpec gridspec,
    const long long u0px, const long long v0px
) {
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
            subgrid[idx] = grid[gridspec.gridToLinear(upx, vpx)];
        }
    }
}

template <typename T, typename S>
auto extractSubgrid(
    const DeviceSpan<T, 2> grid,
    const WorkUnit<S>& workunit,
    const GridConfig gridconf
) {
    // Allocate subgrid matrix
    DeviceArray<ComplexLinearData<S>, 2> subgrid {gridconf.subgrid().shape()};

    auto fn = _extractSubgrid<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgrid.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        subgrid, gridconf.subgrid(), grid, gridconf.padded(), workunit.u0px, workunit.v0px
    );

    return subgrid;
}

template <typename T, typename S>
void degridder(
    HostSpan<WorkUnit<S>*, 1> workunits,
    const DeviceSpan<T, 2> grid,
    const DeviceSpan<S, 2> subtaper,
    const GridConfig gridconf,
    const DegridOp degridop
) {
    // Transfer _unique_ Aterms to GPU
    std::unordered_map<
        std::shared_ptr<HostArray<ComplexLinearData<S>, 2>>,
        const DeviceArray<ComplexLinearData<S>, 2>
    > Aterms;
    for (const auto workunit : workunits) {
        Aterms.try_emplace(workunit->Aleft, *workunit->Aleft);
        Aterms.try_emplace(workunit->Aright, *workunit->Aright);
    }

    // Create and enqueue the work units
    Channel<WorkUnit<S>*> workunitsChannel;
    for (auto workunit : workunits) { workunitsChannel.push(workunit); }
    workunitsChannel.close();

    auto subgridspec = gridconf.subgrid();

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), 4);
        ++i
    ) {
        threads.emplace_back([&] {
            GPU::getInstance().resetDevice(); // needs to be reset for each new thread

            // Make fft plan for each thread
            auto plan = fftPlan<ComplexLinearData<S>>(gridconf.subgrid());

            while (auto maybe = workunitsChannel.pop()) {
                // maybe is a std::optional; let's get the value
                auto workunit = *maybe;

                // Allocate memory for uvdata on device
                DeviceArray<UVDatum<S>, 1> uvdata_d(workunit->data.size());

                // Transfer uvdata data host->device
                bool iscontiguous = workunit->iscontiguous();
                if (iscontiguous) {
                    // If uvdata is sorted, we can avoid a bunch of pointer lookups,
                    // and perform a memcopy on the contiguous memory segment.
                    HostSpan<UVDatum<S>, 1> uvdata_h(
                        {static_cast<long long>(workunit->data.size())},
                        workunit->data.front()
                    );
                    copy(uvdata_d, uvdata_h);
                } else {
                    // Assemble uvdata from (out of order) pointers and transfer to host
                    HostArray<UVDatum<S>, 1> uvdata_h(workunit->data.size());
                    for (size_t i {}; const auto uvdatumptr : workunit->data) {
                        uvdata_h[i++] = *uvdatumptr;
                    }
                    copy(uvdata_d, uvdata_h);
                }

                // Retrieve A terms that have already been sent to device
                const auto& Aleft = Aterms.at(workunit->Aleft);
                const auto& Aright = Aterms.at(workunit->Aright);

                // Allocate subgrid and extract from grid
                auto subgrid = extractSubgrid(grid, *workunit, gridconf);

                // Apply deltal, deltam shift to visibilities
                map([
                    =,
                    deltal=static_cast<S>(subgridspec.deltal),
                    deltam=static_cast<S>(subgridspec.deltam)
                ] __device__ (auto idx, auto& subgrid) {
                    auto [u, v] = subgridspec.linearToUV<S>(idx);
                    subgrid *= cispi(2 * (u * deltal + v * deltam));
                }, Iota(), subgrid);

                fftExec(plan, subgrid, HIPFFT_BACKWARD);

                // Apply aterms, taper and normalize post-FFT
                map([norm = subgrid.size()] __device__ (auto& cell, const auto& Aleft, const auto& Aright, const auto t) {
                    cell = matmul(matmul(Aleft, cell), Aright.adjoint()) *= (t / norm);
                }, subgrid, Aleft, Aright, subtaper);

                gpudft<S>(
                    uvdata_d,
                    {workunit->u0, workunit->v0, workunit->w0},
                    subgrid, gridconf.subgrid(), degridop
                );

                // Transfer data back to host, once again using the optimal strategy
                // depending on whether host data is contiguous
                if (iscontiguous) {
                    HostSpan<UVDatum<S>, 1> uvdata_h(
                        {static_cast<long long>(workunit->data.size())},
                        workunit->data.front()
                    );
                    copy(uvdata_h, uvdata_d);
                } else {
                    HostArray<UVDatum<S>, 1> uvdata_h {uvdata_d};
                    for (size_t i {}; const auto& uvdatum : uvdata_h) {
                        *(workunit->data[i++]) = uvdatum;
                    }
                }
            }

            HIPFFTCHECK( hipfftDestroy(plan) );
        });
    }

    for (auto& t : threads) { t.join(); }
}