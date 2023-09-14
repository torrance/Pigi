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

enum class DegridOp {Replace, Subtract, Add};

template <typename S>
__host__ __device__ inline void degridop_replace(
    ComplexLinearData<S>& olddata, const ComplexLinearData<S>& newdata
) {
    olddata = newdata;
}

template <typename S>
__host__ __device__ inline void degridop_add(
    ComplexLinearData<S>& olddata, const ComplexLinearData<S>& newdata
) {
    olddata += newdata;
}

template <typename S>
__host__ __device__ inline void degridop_subtract(
    ComplexLinearData<S>& olddata, const ComplexLinearData<S>& newdata
) {
    olddata -= newdata;
}

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
    const size_t cachesize {256};
    constexpr int ratio {sizeof(ComplexLinearData<T>) / sizeof(float4)};

    __shared__ float4 _cache[cachesize * ratio];
    auto cache = reinterpret_cast<ComplexLinearData<T>*>(_cache);

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(uvdata.size(), blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        UVDatum<T> uvdatum;
        if (idx < uvdata.size()) uvdatum = uvdata[idx];

        ComplexLinearData<T> data;

        for (size_t i {}; i < subgridspec.size(); i += cachesize) {
            const size_t N = min(cachesize, subgridspec.size() - i);

            // Populate cache
            // We cast to float4 to allow coalesced memory access to global memory,
            // and to avoid bank conflicts when writing to shared memory.
            auto _subgrid = reinterpret_cast<const float4*>(subgrid.data());
            for (size_t j = threadIdx.x, ibase = i * ratio; j < N * ratio; j += blockDim.x) {
                _cache[j] = _subgrid[ibase + j];
            }
            __syncthreads();

            // Cycle through cache
            for (size_t j {}; j < N; ++j) {
                auto [l, m] = subgridspec.linearToSky<T>(i + j);
                auto phase = cispi(-2 * (
                    (uvdatum.u - origin.u0) * l +
                    (uvdatum.v - origin.v0) * m +
                    (uvdatum.w - origin.w0) * ndash(l, m)
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

        switch (degridop) {
        case DegridOp::Replace:
            degridop_replace(uvdatum.data, data);
            break;
        case DegridOp::Add:
            degridop_add(uvdatum.data, data);
            break;
        case DegridOp::Subtract:
            degridop_subtract(uvdatum.data, data);
            break;
        }

        if (idx < uvdata.size()) uvdata[idx] = uvdatum;
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
    int nthreads {256};
    int nblocks = cld<size_t>(uvdata.size(), nthreads);
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
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
    const WorkUnit<S>& workunit
) {
    // Allocate subgrid matrix
    DeviceArray<ComplexLinearData<S>, 2> subgrid {
        workunit.subgridspec.Nx, workunit.subgridspec.Ny
    };

    // Create dummy gridspec to have access to gridToLinear() method
    GridSpec gridspec {grid.size(0), grid.size(1), 0, 0};

    auto fn = _extractSubgrid<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgrid.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        subgrid, workunit.subgridspec, grid, gridspec, workunit.u0px, workunit.v0px
    );

    return subgrid;
}

template <typename T, typename S>
void degridder(
    HostSpan<WorkUnit<S>*, 1> workunits,
    const DeviceSpan<T, 2> grid,
    const DeviceSpan<S, 2> subtaper,
    const DegridOp degridop
) {
    // Transfer Aterms to GPU, since these are often shared
    std::unordered_map<
        const ComplexLinearData<S>*, const DeviceArray<ComplexLinearData<S>, 2>
    > Aterms;
    for (const auto workunit : workunits) {
        Aterms.try_emplace(workunit->Aleft.data(), workunit->Aleft);
        Aterms.try_emplace(workunit->Aright.data(), workunit->Aright);
    }

    // Create and enqueue the work units
    Channel<WorkUnit<S>*> workunitsChannel;
    for (auto workunit : workunits) { workunitsChannel.push(workunit); }
    workunitsChannel.close();

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), std::thread::hardware_concurrency());
        ++i
    ) {
        threads.emplace_back([&] {
            // Make fft plan for each thread
            auto plan = fftPlan<ComplexLinearData<S>>(workunits.front()->subgridspec);

            while (auto maybe = workunitsChannel.pop()) {
                // maybe is a std::optional; let's get the value
                auto workunit = *maybe;

                // Transfer data to device and retrieve A terms
                DeviceArray<UVDatum<S>, 1> uvdata {workunit->data};
                const auto& Aleft = Aterms.at(workunit->Aleft.data());
                const auto& Aright = Aterms.at(workunit->Aright.data());

                // Allocate subgrid and extract from grid
                auto subgrid = extractSubgrid(grid, *workunit);

                fftExec(plan, subgrid, HIPFFT_BACKWARD);

                // Apply aterms, taper and normalize post-FFT
                map([norm = subgrid.size()] __device__ (auto& cell, const auto& Aleft, const auto& Aright, const auto t) {
                    cell = matmul(matmul(Aleft, cell), Aright.adjoint()) *= (t / norm);
                }, subgrid, Aleft, Aright, subtaper);

                gpudft<S>(
                    uvdata,
                    {workunit->u0, workunit->v0, workunit->w0},
                    subgrid, workunit->subgridspec, degridop
                );

                // Transfer data back to host
                workunit->data = uvdata;
            }

            HIPFFTCHECK( hipfftDestroy(plan) );
        });
    }

    for (auto& t : threads) { t.join(); }
}