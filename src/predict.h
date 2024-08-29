#pragma once

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "aterms.h"
#include "channel.h"
#include "datatable.h"
#include "degridder.h"
#include "fft.h"
#include "hip.h"
#include "memory.h"
#include "splitter.h"
#include "taper.h"
#include "timer.h"
#include "util.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
void predict(
    DataTable& tbl,
    std::vector<WorkUnit>& workunits,
    const HostSpan<T<S>, 2> img,
    const GridConfig gridconf,
    Aterms& aterms,
    const DegridOp degridop
) {
    auto timer = Timer::get("predict");

    const auto gridspec = gridconf.padded();
    const auto subgridspec = gridconf.subgrid();

    // Copy img into wlayer on device. wlayer is shared by all threads to populate
    // subgrid visibilities. wlayer is shared since it may be very large.
    DeviceArray<T<S>, 2> wlayer(img);
    double wold {0};  // track the associated w value of w-layer

    // Use a lock_guard to guarantee exclusive access during wlayer processing.
    std::mutex wlock;

    // Add padding if required
    if (gridconf.grid() != gridspec) wlayer = resize(wlayer, gridconf.grid(), gridspec);

    // Apply taper
    {
        auto timer = Timer::get("predict::taper");
        DeviceArray<S, 2> taperd {pswf<S>(gridspec)};
        map([] __device__ (auto& wlayer, const auto t) {
            if (t == 0) wlayer = T<S>{};
            else wlayer /= t;
        }, wlayer, taperd);
    }

    // Create FFT plan for wlayer
    auto wplan = fftPlan<T<S>>(gridspec);

    // Copy subtaper to device
    DeviceArray<S, 2> subtaper_d {pswf<S>(subgridspec)};

    // Lambdas does not change row to row; send to device now
    const DeviceArray<double, 1> lambdas_d(tbl.lambdas());

    // Start by precomputing batch boundaries and loading up the channel
    // used by the threads to determine their scope of work.
    Channel<std::array<size_t, 4>> batches;
    {
        size_t maxmem = [] () -> size_t {
            size_t free, total;
            HIPCHECK( hipMemGetInfo(&free, &total));
            return (free * 0.8) / 2;
        }();
        Logger::debug(
            "Setting maxmem per thread to {:.3f} GB",
            maxmem / 1024. / 1024. / 1024.
        );
        if (maxmem < 1024llu * 1024 * 1024) Logger::warning(
            "Memory per thead is less than 1 GB (free {:.3f} GB)",
            maxmem / 1024. / 1024. / 1024.
        );

        // Precompute memory
        // TODO: add mem usage of Aterms
        size_t workunitmem = (
            sizeof(WorkUnit) +
            gridconf.subgrid().size() * sizeof(T<S>)
        );
        size_t rowmem = (
            sizeof(DataTable::RowMetadata) +
            sizeof(std::array<double, 3>) +
            tbl.nchans() * sizeof(ComplexLinearData<float>)
        );

        // Estimate memory used in subgrid FFT. In the worst case, mem usage grows
        // linearly with the batch size (observed with AMD). Otherwise, it's approximately
        // fixed, and is neglible anyway (observed with CUDA).
        size_t fftmem = fftEstimate<T<S>>(subgridspec, workunits.size()) / workunits.size();

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
            size_t mem = nworkunits * (workunitmem + fftmem) + nrows * rowmem;

            if (
                mem > maxmem ||                       // maximum batch size
                wkend == workunits.size()             // final iteration
            ) {
                batches.push({wkstart, wkend, rowstart, rowend});
                wkstart = wkend;
            }
        }
    }
    batches.close();

    // Create the threads
    std::vector<std::thread> threads;
    for (size_t threadid {}; threadid < 2; ++threadid) {
        threads.emplace_back([&] {
            GPU::getInstance().resetDevice(); // needs to be reset for each new thread
            auto timer = Timer::get("predict::batch");

            while (auto batch = batches.pop()) {
                auto& [wkstart, wkend, rowstart, rowend] = *batch;
                long long nworkunits = wkend - wkstart;
                long long nrows = rowend - rowstart;

                Logger::debug(
                    "Invert: batching rows {}-{}/{} ({} workunits)",
                    rowstart, rowend, workunits.size(), nworkunits
                );

                // Allocate workunits
                HostSpan<WorkUnit, 1> workunits_h({nworkunits}, workunits.data() + wkstart);
                DeviceArray<WorkUnit, 1> workunits_d(workunits_h);

                // Allocate subgrid stack
                DeviceArray<T<S>, 3> subgrids_d({subgridspec.Nx, subgridspec.Ny, nworkunits});

                // Group subgrids into w layers
                std::unordered_map<double, std::vector<size_t>> widxs;
                for (size_t i {wkstart}; i < wkend; ++i) {
                    auto workunit = workunits[i];
                    widxs[workunit.w].push_back(i - wkstart);
                }

                // ...and process each wlayer serially
                for (std::lock_guard l(wlock); auto& [w0, idxs] : widxs) {
                    auto timer = Timer::get("predict::wlayers");

                    HIPFFTCHECK( hipfftSetStream(wplan, hipStreamPerThread) );

                    // Move wlayer from w=wold -> w=w0
                    PIGI_TIMER(
                        "predict::wlayers::wdecorrection",
                        map([gridspec, w0=static_cast<S>(w0), wold=static_cast<S>(wold)] __device__ (
                            auto idx, auto& wlayer
                        ) {
                            auto [l, m] = gridspec.linearToSky<S>(idx);
                            wlayer *= cispi(-2 * (w0 - wold) * ndash(l, m));
                        }, Iota(), wlayer)
                    );

                    // Set wold to new w value
                    wold = w0;

                    // FFT wlayer from sky -> visibility domain
                    PIGI_TIMER(
                        "predict::wlayers::fft",
                        fftExec(wplan, wlayer, HIPFFT_FORWARD);
                    );

                    // Populate subgrid stack with subgrids from this wlayer
                    splitter(
                        subgrids_d, wlayer, DeviceArray<size_t, 1>(idxs),
                        workunits_d, gridspec, subgridspec
                    );

                    // FFT wlayer from visibility -> sky domain
                    PIGI_TIMER(
                        "predict::wlayers::fft",
                        fftExec(wplan, wlayer, HIPFFT_BACKWARD);
                    );

                    // Normalize the inverse FFT
                    map([N=gridspec.size()] __device__ (auto& wlayer) {
                        wlayer /= N;
                    }, wlayer);

                    // Ensure all work is complete before releasing the wlock
                    HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
                }  // loop: wlayers

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

                HostArray<std::array<double, 3>, 1> uvws_h(nrows);
                for (size_t i {}; auto m : tbl.metadata({rowstart, rowend})) {
                    uvws_h[i++] = {m.u, m.v, m.w};
                }

                /// Timer is heap allocated here so we can deallocate manually
                Timer::StopWatch* timer = new Timer::StopWatch;
                *timer = Timer::get("predict::batch::memHtoD");

                // Copy across data
                DeviceArray<ComplexLinearData<float>, 2> data_d(data_h);
                DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);
                DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts_d(alefts_h);
                DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights_d(arights_h);

                delete timer;

                // Apply deltal, deltam shift to visibilities
                PIGI_TIMER(
                    "predict::batch::subgridsdeltalm",
                    map([
                        =,
                        deltal=static_cast<S>(subgridspec.deltal),
                        deltam=static_cast<S>(subgridspec.deltam),
                        stride=subgridspec.size()
                    ] __device__ (auto idx, auto& subgrid) {
                        idx %= stride;
                        auto [u, v] = subgridspec.linearToUV<S>(idx);
                        subgrid *= cispi(2 * (u * deltal + v * deltam));
                    }, Iota(), subgrids_d);
                );

                // Shift to sky domain
                PIGI_TIMER(
                    "predict::batch::subgridfft",
                    auto subplan = fftPlan<T<S>>(subgridspec, nworkunits);
                    fftExec(subplan, subgrids_d, HIPFFT_BACKWARD);
                    hipfftDestroy(subplan);
                );

                // Degrid
                degridder<T<S>, S>(
                    data_d, subgrids_d, workunits_d, uvws_d, lambdas_d, subtaper_d,
                    alefts_d, arights_d, subgridspec, rowstart, degridop
                );

                // Copy data back to host
                PIGI_TIMER(
                    "predict::batch::memDtoH",
                    copy(data_h, data_d)
                );
            }  // loop: batches
        });
    } // loop: threads

    // Wait for all threads to complete
    for (auto& thread : threads) thread.join();
    hipfftDestroy(wplan);
}