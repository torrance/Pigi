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
#include "timer.h"
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

        // Convert to wavenumbers
        l *= 2 * ::pi_v<S>;
        m *= 2 * ::pi_v<S>;
        n *= 2 * ::pi_v<S>;

        ComplexLinearData<S> cell {};

        for (
            size_t i {blockIdx.y * cachesize};
            i < uvdata.size();
            i += (gridDim.y * cachesize)
        ) {
            const size_t N = min(cachesize, uvdata.size() - i);

            // Populate cache
            __syncthreads();
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

            // Zombie threads need to keep filling the cache
            // but can skip doing any actual work
            if (idx >= subgridspec.size()) continue;

            // Read through cache
            for (size_t j {}; j < N; ++j) {
                // Retrieve value of uvdatum from the cache
                // This shared mem load is broadcast across the warp and so we
                // don't need to worry about bank conflicts
                UVDatum<S> uvdatum = cache[j];

                auto phase = cis(uvdatum.u * l + uvdatum.v * m + uvdatum.w * n);

                // Equivalent of: cell += uvdata.data * phase
                // Written out explicitly to use fma operations
                cmac(cell.xx, uvdatum.data.xx, phase);
                cmac(cell.yx, uvdatum.data.yx, phase);
                cmac(cell.xy, uvdatum.data.xy, phase);
                cmac(cell.yy, uvdatum.data.yy, phase);
            }
        }

        // Zombie threads can exit early
        if (idx >= subgridspec.size()) return;

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

        atomicAdd(subgrid.data() + idx, output);
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
    auto timer = Timer::get("invert::wlayer::gridder::thread::dift");

    // x-dimension corresponds to cells in the subgrid
    int nthreadsx {128}; // hardcoded to match the cache size
    int nblocksx = cld<size_t>(subgridspec.size(), nthreadsx);

    // y-dimension breaks down chunks of uvdata necessary to increase occupancy
    // but we cap the blocks at 12 which already gives good occupancy
    int nthreadsy {1};
    int nblocksy = std::min(12UL, cld<size_t>(uvdata.size(), nthreadsx));

    if (makePSF) {
        auto fn = _gpudift<T, S, true>;
        hipLaunchKernelGGL(
            fn, dim3(nblocksx, nblocksy, 1), dim3(nthreadsx, nthreadsy, 1),
            0, hipStreamPerThread,
            subgrid, Aleft, Aright,
            origin, uvdata, subgridspec
        );
    } else {
        auto fn = _gpudift<T, S, false>;
        hipLaunchKernelGGL(
            fn, dim3(nblocksx, nblocksy, 1), dim3(nthreadsx, nthreadsy, 1),
            0, hipStreamPerThread,
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
    const long long u0px, const long long v0px
) {
    auto timer = Timer::get("invert::wlayer::gridder::thread::adder");

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
    const GridConfig gridconf,
    const bool makePSF
) {
    auto timer = Timer::get("invert::wlayer::gridder");

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
    Channel<Pair> subgridsChannel;

    // Enqueue the work units
    for (const auto workunit : workunits) { workunitsChannel.push(workunit); }
    workunitsChannel.close();

    // Ensure all memory transfers have completed before spawning theads
    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

    std::vector<std::thread> threads;
    for (
        size_t i {};
        i < std::min<size_t>(workunits.size(), 4);
        ++i
    ) {
        threads.emplace_back([&] {
            GPU::getInstance().resetDevice(); // needs to be reset for each new thread
            auto timer = Timer::get("invert::wlayer::gridder::thread");

            // Make FFT plan for each thread
            auto plan = fftPlan<T>(subgridspec);

            while (auto  maybe = workunitsChannel.pop()) {
                // maybe is a std::optional; let's get the value
                const auto workunit = *maybe;

                const UVWOrigin origin {workunit->u0, workunit->v0, workunit->w0};

                // Transfer workunit pointers to device and allocate uvdata memory
                DeviceArray<UVDatum<S>*, 1> uvdata_ptrs(workunit->data);
                DeviceArray<UVDatum<S>, 1> uvdata(workunit->data.size());

                // Dereference uvdata prior to the gridding kernel
                PIGI_TIMER(
                    "invert::wlayer::gridder::thread::memcoalesce",
                    map([] __device__ (auto& uvdatum, auto ptr) {
                        uvdatum = *ptr;
                    }, uvdata, uvdata_ptrs);
                );

                // Retrieve A terms that have already been sent to device
                const auto& Aleft = Aterms.at(workunit->Aleft);
                const auto& Aright = Aterms.at(workunit->Aright);

                // Allocate subgrid
                DeviceArray<T, 2> subgrid {subgridspec.Nx, subgridspec.Ny};

                // DFT
                gpudift<T, S>(
                    subgrid, Aleft, Aright, origin, uvdata, subgridspec, makePSF
                );

                // Apply taper and perform FFT normalization
                PIGI_TIMER(
                    "invert::wlayer::gridder::thread::prefft",
                    map([N = subgrid.size()] __device__ (auto& cell, const auto t) {
                        cell *= (t / N);
                    }, subgrid, subtaper);
                );

                // FFT
                PIGI_TIMER("invert::wlayer::gridder::thread::fft", fftExec(plan, subgrid, HIPFFT_FORWARD));

                // Reset deltal, deltam shift prior to adding to master grid
                PIGI_TIMER(
                    "invert::wlayer::gridder::thread::deltlm",
                    map([
                        =,
                        deltal=static_cast<S>(subgridspec.deltal),
                        deltam=static_cast<S>(subgridspec.deltam)
                    ] __device__ (auto idx, auto& subgrid) {
                        auto [u, v] = subgridspec.linearToUV<S>(idx);
                        subgrid *= cispi(-2 * (u * deltal + v * deltam));
                    }, Iota(), subgrid);
                );

                // Sync the stream before we send the subgrid back to the main thread
                HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

                subgridsChannel.push(Pair(std::move(subgrid), workunit));
            }

            HIPFFTCHECK( hipfftDestroy(plan) );
        });
    }

    // Meanwhile, process subgrids and add back to the main grid as the become available
    for (size_t i {}; i < workunits.size(); ++i) {
        const auto [subgrid, workunit] = subgridsChannel.pop().value();
        addsubgrid<T>(grid, subgrid, gridspec, subgridspec, workunit->u0px, workunit->v0px);
    }

    for (auto& t : threads) { t.join(); }
}