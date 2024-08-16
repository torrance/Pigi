#include <algorithm>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "aterms.h"
#include "channel.h"
#include "datatable.h"
#include "fft.h"
#include "hip.h"
#include "memory.h"
#include "taper.h"
#include "timer.h"
#include "util.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invert(
    DataTable& tbl,
    std::vector<WorkUnit>& workunits,
    GridConfig gridconf,
    Aterms& aterms,
    bool makePSF = false
) {
    auto timer = Timer::get("invert");

    GridSpec gridspec = gridconf.padded();
    GridSpec subgridspec = gridconf.subgrid();

    // Allocate img, wlayer which will be summed by each thread
    DeviceArray<T<S>, 2> imgd(gridspec.Nx, gridspec.Ny);
    DeviceArray<T<S>, 2> wlayer(gridspec.Nx, gridspec.Ny);

    // Create FFT plan for wlayer
    auto wplan = fftPlan<T<S>>(gridspec);

    // Construct the taper and send to the device
    DeviceArray<S, 2> subtaperd {pswf<S>(subgridspec)};

    // Lambdas does not change row to row; send to device now
    const DeviceArray<double, 1> lambdas_d(tbl.lambdas());

    // Create channels used for sending work to threads, and for recieving
    // completed subgrid stacks into the parent thread

    // std::array<wkstart, wkend, rowstart, rowend>
    Channel<std::array<size_t, 4>> batches;

    // std::array<wkstart, wkend, workunits_d, subgrids_d>
    // Note this has a buffer size of 1, ensuring only (nthreads + 1) subgrid stacks exist.
    Channel<
        std::tuple<size_t, size_t, DeviceArray<WorkUnit, 1>, DeviceArray<T<S>, 3>>
    > stacks(1);

    // Compute batch boundaries and load up the batches channel
    // used by the threads to determine their scope of work.
    {
        size_t maxmem = [] () -> size_t {
            size_t free, total;
            HIPCHECK( hipMemGetInfo(&free, &total));
            return (free * 0.8) / 2;
        }();
        Logger::debug(
            "Setting maxmem per thread to {} GB",
            maxmem / 1024. / 1024. / 1024.
        );

        // Precompute memory
        size_t workunitmem = (
            sizeof(WorkUnit) +
            subgridspec.size() * sizeof(T<S>)
        );
        size_t rowmem = (
            sizeof(DataTable::RowMetadata) +
            sizeof(std::array<double, 3>) +  // uvws
            tbl.nchans() * (
                sizeof(ComplexLinearData<float>) +  // data
                sizeof(LinearData<float>)           // weights
            )
        );

        // Loop state
        for (size_t wkstart {}, wkend {1}; wkend <= workunits.size(); ++wkend) {
            // Align wkend with a new row
            while (
                wkend <= workunits.size() && workunits[wkend - 1].chanend < tbl.nchans()
            ) ++wkend;

            size_t rowstart = workunits[wkstart].rowstart;
            size_t rowend = workunits[wkend - 1].rowend;

            size_t nrows = rowend - rowstart;
            size_t nworkunits = wkend - wkstart;

            // Calculate size of current bounds
            // The factor of 2 accounts for the extra subgrids that might be held in memory
            // by the main thread for adding.
            size_t mem = 2 * nworkunits * workunitmem + nrows * rowmem;

            if (
                mem > maxmem ||                       // maximum batch size
                nworkunits > workunits.size() / 2 ||  // require >= 2 batches minimum
                wkend == workunits.size()             // final iteration
            ) {
                batches.push({wkstart, wkend, rowstart, rowend});
                wkstart = wkend;
            }
        }
        batches.close();
    }

    std::vector<std::thread> threads;
    for (size_t threadid {}; threadid < 2; ++threadid) {
        threads.emplace_back([&] {
            GPU::getInstance().resetDevice(); // needs to be reset for each new thread
            auto timer = Timer::get("invert::batch");

            // Now loop over the batches until they are exhausted
            while (auto batch = batches.pop()) {
                auto [wkstart, wkend, rowstart, rowend] = *batch;
                long long nworkunits = wkend - wkstart;
                long long nrows = rowend - rowstart;

                Logger::debug(
                    "Invert: batching rows {}-{}/{} ({} workunits)",
                    rowstart, rowend, workunits.size(), nworkunits
                );

                HostSpan<WorkUnit, 1> workunits_h({nworkunits}, workunits.data() + wkstart);

                // Create aterms arrays
                HostArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts_h(nworkunits);
                HostArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights_h(nworkunits);

                // Now transfer across all required Aterms and update the Aleft, Aright values
                // in workunits_h. We use a dictionary copy across only _unique_ Aterms, since
                // these may be shared across workunits.
                std::unordered_map<
                    Aterms::aterm_t, DeviceArray<ComplexLinearData<double>, 2>
                > aterm_map;
                for (size_t i {}; auto w : workunits_h) {
                    auto [ant1, ant2] = w.baseline;

                    Aterms::aterm_t aleft = aterms.get(w.time, ant1);
                    alefts_h[i] = (*aterm_map.try_emplace(aleft, *aleft).first).second;

                    Aterms::aterm_t aright = aterms.get(w.time, ant2);
                    arights_h[i] = (*aterm_map.try_emplace(aright, *aright).first).second;

                    ++i;
                }

                auto data_h = tbl.data({rowstart, rowend});
                auto weights_h = tbl.weights({rowstart, rowend});

                HostArray<std::array<double, 3>, 1> uvws_h(nrows);
                for (size_t i {}; auto m : tbl.metadata({rowstart, rowend})) {
                    uvws_h[i++] = {m.u, m.v, m.w};
                }

                // Copy across data
                DeviceArray<WorkUnit, 1> workunits_d(workunits_h);
                DeviceArray<ComplexLinearData<float>, 2> data_d(data_h);
                DeviceArray<LinearData<float>, 2> weights_d(weights_h);
                DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);
                DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts_d(alefts_h);
                DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights_d(arights_h);

                // Allocate subgrid stack
                DeviceArray<T<S>, 3> subgrids_d({subgridspec.Nx, subgridspec.Ny, nworkunits});

                // Grid
                gridder<T<S>, S>(
                    subgrids_d, workunits_d, uvws_d, data_d, weights_d,
                    subgridspec, lambdas_d, subtaperd, alefts_d, arights_d, rowstart, makePSF
                );

                // FFT all subgrids as one batch
                PIGI_TIMER(
                    "invert::batch::subgridfft",
                    auto subplan = fftPlan<T<S>>(subgridspec, nworkunits);
                    fftExec(subplan, subgrids_d, HIPFFT_FORWARD);
                    hipfftDestroy(subplan);
                );

                // Reset deltal, deltam shift prior to adding to master grid
                PIGI_TIMER(
                    "invert::batch::subgridsdeltalm",
                    map([
                        =,
                        deltal=static_cast<S>(subgridspec.deltal),
                        deltam=static_cast<S>(subgridspec.deltam),
                        stride=subgridspec.size()
                    ] __device__ (auto idx, auto& px) {
                        idx %= stride;
                        auto [u, v] = subgridspec.linearToUV<S>(idx);
                        px *= cispi(-2 * (u * deltal + v * deltam));
                    }, Iota(), subgrids_d)
                );

                // Send the processed subgrids off to the main thread to add into
                // wlayer and imgd
                stacks.push(
                    {wkstart, wkend, std::move(workunits_d), std::move(subgrids_d)}
                );
            }  // loop: batches
        });  // thread lambda functions
    }  // loop: threads

    // In the main thread, process subgrid stacks and append to img
    for (size_t nworkunitsdone {}; nworkunitsdone < workunits.size();) {
        auto stack = stacks.pop();
        auto& [wkstart, wkend, workunits_d, subgrids_d] = *stack;
        nworkunitsdone += workunits_d.size();  // termination condiation for this loop

        // Start the timer _after_ we've recieved the stack, otherwise we're just
        // measuring the wait time on the channel
        auto timer = Timer::get("invert::wlayers");

        // Group subgrids into w layers
        std::unordered_map<double, std::vector<size_t>> widxs;
        for (size_t i {wkstart}; i < wkend; ++i) {
            auto workunit = workunits[i];
            widxs[workunit.w].push_back(i - wkstart);
        }

        // ...and process each wlayer serially
        for (auto& [w0, idxs] : widxs) {
            wlayer.zero();

            // Add each subgrid from this w-layer
            addsubgrid(
                wlayer, DeviceArray<size_t, 1>(idxs),
                workunits_d, subgrids_d, gridspec, subgridspec
            );

            // Apply deltal, deltam shift
            PIGI_TIMER(
                "invert::wlayers::deltalm",
                map([
                    =,
                    deltal=static_cast<S>(gridspec.deltal),
                    deltam=static_cast<S>(gridspec.deltam)
                ] __device__ (auto idx, auto& wlayer) {
                    auto [u, v] = gridspec.linearToUV<S>(idx);
                    wlayer *= cispi(2 * (u * deltal + v * deltam));
                }, Iota(), wlayer)
            );

            // FFT the full wlayer
            PIGI_TIMER(
                "invert::wlayers::fft",
                fftExec(wplan, wlayer, HIPFFT_BACKWARD)
            );

            // Apply wcorrection and append layer onto img
            PIGI_TIMER(
                "invert::wlayers::wcorrection",
                map([gridspec=gridspec, w0=static_cast<S>(w0)] __device__ (
                    auto idx, auto& imgd, auto wlayer
                ) {
                    auto [l, m] = gridspec.linearToSky<S>(idx);
                    wlayer *= cispi(2 * w0 * ndash(l, m));
                    imgd += wlayer;
                }, Iota(), imgd, wlayer)
            );
        }  // loop: wlayers
    }

    hipfftDestroy(wplan);

    // Wait for all threads to complete
    for (auto& thread : threads) thread.join();

    // The final image still has a taper applied. It's time to remove it.
    map([] __device__ (auto& px, auto t) {
        px /= t;
    }, imgd, DeviceArray<S, 2>(pswf<S>(gridspec)));

    // Copy image from device to host
    HostArray<T<S>, 2> img(imgd);

    return resize(img, gridconf.padded(), gridconf.grid());
}

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
    size_t rowoffset,
    bool makePSF
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<float>) + sizeof(S))
    ];
    auto data_cache = reinterpret_cast<ComplexLinearData<float>*>(_cache);
    auto invlambdas_cache = reinterpret_cast<S*>(data_cache + cachesize);

    // Get workunit information
    size_t rowstart, rowend, chanstart, chanend;
    S u0, v0, w0;
    {
        const auto& workunit = workunits[blockIdx.y];
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
            const auto Al = static_cast<ComplexLinearData<S>>(alefts[blockIdx.y][idx]).inv();
            const auto Ar = static_cast<ComplexLinearData<S>>(arights[blockIdx.y][idx]).inv().adjoint();

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

        subgrids[blockIdx.y * subgridspec.size() + idx] = output;
    }
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
    size_t rowoffset,
    bool makePSF
) {
    auto timer = Timer::get("invert::batch::gridder");

    // x-dimension corresponds to cells in the subgrid
    int nthreadsx {128}; // hardcoded to match the cache size
    int nblocksx = cld<size_t>(subgridspec.size(), nthreadsx);

    // y-dimension corresponds to workunit index
    int nthreadsy {1};
    int nblocksy = workunits.size();

    auto fn = _gridder<T, S>;
    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy, 1), dim3(nthreadsx, nthreadsy, 1),
        0, hipStreamPerThread,
        subgrids, workunits, uvws, data, weights, subgridspec, lambdas,
        subtaper, alefts, arights, rowoffset, makePSF
    );
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void _addsubgrid(
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
void addsubgrid(
    DeviceSpan<T, 2> grid, const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits, const DeviceSpan<T, 3> subgrids,
    const GridSpec gridspec, const GridSpec subgridspec
) {
    auto timer = Timer::get("invert::wlayers::adder");

    auto fn = _addsubgrid<T>;
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