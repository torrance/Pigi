#pragma once

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "aterms.h"
#include "datatable.h"
#include "fft.h"
#include "hip.h"
#include "memory.h"
#include "taper.h"
#include "timer.h"
#include "util.h"
#include "workunit.h"

enum class DegridOp {Subtract, Add};

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

    // Copy img (with padding) to device and apply inverse taper
    DeviceArray<T<S>, 2> imgd {resize(img, gridconf.grid(), gridspec)};
    {
        DeviceArray<S, 2> taperd {pswf<S>(gridspec)};
        map([] __device__ (auto& img, const auto t) {
            if (t == 0) img = T<S>{};
            else img /= t;
        }, imgd, taperd);
    }

    // Copy subtaper to device
    DeviceArray<S, 2> subtaper_d {pswf<S>(subgridspec)};

    // Create wlayer on device
    DeviceArray<T<S>, 2> wlayer {gridspec.shape(), false};

    // TODO: set batch dynamically, based on memory allowance
    const size_t nbatch {4096};

    // Create FFT plans
    auto wplan = fftPlan<T<S>>(gridspec);
    auto subplan = fftPlan<T<S>>(subgridspec, nbatch);

    // Lambdas does not change row to row; send to device now
    const DeviceArray<double, 1> lambdas_d(tbl.lambdas());

    for (size_t wkstart {}; wkstart < workunits.size(); wkstart += nbatch) {
        auto timer = Timer::get("predict::batch");

        // wkstart, wkend are the bounds of the workunits to be used this batch
        size_t wkend = std::min(wkstart + nbatch, workunits.size());
        long long nworkunits = wkend - wkstart;

        // Record the corresponding row dimensions of this batch
        size_t rowstart = workunits[wkstart].rowstart;
        size_t rowend = workunits[wkend - 1].rowend;
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

        HostArray<std::array<double, 3>, 1> uvws_h(nrows);
        for (size_t i {}; auto m : tbl.metadata({rowstart, rowend})) {
            uvws_h[i++] = {m.u, m.v, m.w};
        }

        // Copy across data
        DeviceArray<WorkUnit, 1> workunits_d(workunits_h);
        DeviceArray<ComplexLinearData<float>, 2> data_d(data_h);
        DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);
        DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts_d(alefts_h);
        DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights_d(arights_h);

        // Allocate subgrid stack
        DeviceArray<T<S>, 3> subgrids_d({subgridspec.Nx, subgridspec.Ny, nbatch});

        // Group subgrids into w layers
        std::unordered_map<double, std::vector<size_t>> widxs;
        for (size_t i {wkstart}; i < wkend; ++i) {
            auto workunit = workunits[i];
            widxs[workunit.w].push_back(i - wkstart);
        }

        // ...and process each wlayer serially
        for (auto& [w0, idxs] : widxs) {
            auto timer = Timer::get("predict::batch::wlayers");
            wlayer.zero();

            // Apply w-decorrection to img and copy to wlayer
            PIGI_TIMER(
                "predict::batch::wlayers::wdecorrection",
                map([w0=w0, gridspec=gridspec] __device__ (auto idx, auto img, auto& wlayer) {
                    auto [l, m] = gridspec.linearToSky<S>(idx);
                    img *= cispi(-2 * w0 * ndash(l, m));
                    wlayer = img;
                }, Iota(), imgd, wlayer)
            );

            // Transform from sky => visibility domain
            PIGI_TIMER(
                "predict::batch::wlayers::fft",
                fftExec(wplan, wlayer, HIPFFT_FORWARD)
            );

            // Reset deltal, deltam shift to visibilities
            PIGI_TIMER(
                "predict::batch::wlayers::deltalm",
                map([
                    =,
                    deltal=static_cast<S>(gridspec.deltal),
                    deltam=static_cast<S>(gridspec.deltam)
                ] __device__ (auto idx, auto& wlayer) {
                    auto [u, v] = gridspec.linearToUV<S>(idx);
                    wlayer *= cispi(-2 * (u * deltal + v * deltam));
                }, Iota(), wlayer)
            );

            // Populate subgrid stack with subgrids from this wlayer
            extractSubgrid<T<S>>(
                subgrids_d, wlayer, DeviceArray<size_t, 1>(idxs),
                workunits_d, gridspec, subgridspec
            );
        }

        // Now the subgrid is loaded up, we can proceed with degridding

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
            fftExec(subplan, subgrids_d, HIPFFT_BACKWARD)
        );

        // Degrid
        degridder<T<S>, S>(
            data_d, subgrids_d, workunits_d, uvws_d, lambdas_d, subtaper_d,
            alefts_d, arights_d, subgridspec, rowstart, degridop
        );

        // Copy data back to host
        copy(data_h, data_d);
    }
}

template <typename T, typename S>
__global__
void _degridder(
    DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const GridSpec subgridspec,
    size_t rowoffset,
    const DegridOp degridop
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[
        cachesize * (sizeof(ComplexLinearData<S>) + sizeof(std::array<S, 3>))
    ];
    auto subgrid_cache = reinterpret_cast<ComplexLinearData<S>*>(_cache);
    auto lmn_cache = reinterpret_cast<std::array<S, 3>*>(subgrid_cache + cachesize);

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

    const size_t nchans = chanend - chanstart;
    const size_t nvis = (rowend - rowstart) * nchans;
    const size_t rowstride = lambdas.size();

    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < blockDim.x * cld<size_t>(nvis, blockDim.x);
        idx += blockDim.x * gridDim.x
    ) {
        size_t irow = idx / nchans + rowstart;
        size_t ichan = idx % nchans + chanstart;

        // Get metadata for this datum
        S lambda;
        std::array<double, 3> uvw;
        if (idx < nvis) {
            lambda = lambdas[ichan];
            uvw = uvws[irow];
        }

        // Precompute uvw offsets and convert to wavenumbers
        S u = -2 * ::pi_v<S> * (std::get<0>(uvw) / lambda - u0);
        S v = -2 * ::pi_v<S> * (std::get<1>(uvw) / lambda - v0);
        S w = -2 * ::pi_v<S> * (std::get<2>(uvw) / lambda - w0);

        ComplexLinearData<S> datum;

        for (size_t ipx {}; ipx < subgridspec.size(); ipx += cachesize) {
            const size_t N = min(cachesize, subgridspec.size() - ipx);

            // Populate cache
            __syncthreads();
            for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                // Load subgrid value; convert to instrumental values
                auto cell = static_cast<ComplexLinearData<S>>(
                    subgrids[subgridspec.size() * blockIdx.y + ipx + j]
                );

                // Grab A terms
                auto al = static_cast<ComplexLinearData<S>>(alefts[blockIdx.y][ipx + j]);
                auto ar = static_cast<ComplexLinearData<S>>(arights[blockIdx.y][ipx + j]).adjoint();

                // Apply Aterms, normalization, and taper
                cell = matmul(matmul(al, cell), ar);
                cell *= subtaper[ipx + j] / subgridspec.size();

                subgrid_cache[j] = cell;

                // Precompute l, m, n and cache values
                auto [l, m] = subgridspec.linearToSky<S>(ipx + j);
                auto n = ndash(l, m);

                lmn_cache[j] = {l, m, n};
            }
            __syncthreads();

            // Zombie threads need to keep filling the cache
            // but can skip doing any actual work
            if (idx >= nvis) continue;

            // Cycle through cache
            for (size_t j {}; j < N; ++j) {
                auto [l, m, n] = lmn_cache[j];
                auto phase = cis(u * l + v * m + w * n);

                // Load subgrid cell from the cache
                // This shared mem load is broadcast across the warp and so we
                // don't need to worry about bank conflicts
                auto cell = subgrid_cache[j];

                // Equivalent of: data += cell * phase
                // Written out explicitly to use fma operations
                cmac(datum.xx, cell.xx, phase);
                cmac(datum.yx, cell.yx, phase);
                cmac(datum.xy, cell.xy, phase);
                cmac(datum.yy, cell.yy, phase);
            }
        }

        if (idx < nvis) {
            switch (degridop) {
            case DegridOp::Add:
                data[irow * rowstride + ichan]
                    += static_cast<ComplexLinearData<float>>(datum);
                break;
            case DegridOp::Subtract:
                data[irow * rowstride + ichan]
                    -= static_cast<ComplexLinearData<float>>(datum);
                break;
            }
        }
    }
}

template <typename T, typename S>
void degridder(
    DeviceSpan<ComplexLinearData<float>, 2> data,
    const DeviceSpan<T, 3> subgrids,
    const DeviceSpan<WorkUnit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts,
    const DeviceSpan<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights,
    const GridSpec subgridspec,
    size_t rowoffset,
    const DegridOp degridop
) {
    auto timer = Timer::get("predict::batch::degridder");

    auto fn = _degridder<T, S>;

    // x-dimension distributes uvdata
    int nthreadsx {128};
    int nblocksx {1};

    // y-dimension breaks the subgrid down into 8 blocks
    int nthreadsy {1};
    int nblocksy = workunits.size();

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        data, subgrids, workunits, uvws, lambdas, subtaper, alefts, arights, subgridspec,
        rowoffset, degridop
    );
}

template <typename T>
__global__
void _extractSubgrid(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<T, 2> grid,
    const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits,
    const GridSpec gridspec,
    const GridSpec subgridspec
) {
    size_t widx = widxs[blockIdx.y];
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
            subgrids[subgridspec.size() * widx + idx]
                = grid[gridspec.gridToLinear(upx, vpx)];
        }
    }
}

template <typename T>
auto extractSubgrid(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<T, 2> grid,
    const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<WorkUnit, 1> workunits,
    const GridSpec gridspec,
    const GridSpec subgridspec
) {
    auto timer = Timer::get("predict::batch::wlayers::splitter");

    auto fn = _extractSubgrid<T>;
    auto [nblocksx, nthreadsx] = getKernelConfig(
        fn, subgridspec.size()
    );

    int nthreadsy {1};
    int nblocksy = widxs.size();

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        subgrids, grid, widxs, workunits, gridspec, subgridspec
    );
}