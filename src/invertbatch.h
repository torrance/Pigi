#include <vector>

#include <hip/hip_runtime.h>

#include "memory.h"
#include "visibility.h"

template <typename T, typename S>
__global__
void _gridderk(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<Visibility::Workunit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<ComplexLinearData<S>, 2> data,
    const DeviceSpan<LinearData<S>, 2> weights,
    const GridSpec subgridspec,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    size_t rowoffset,
    bool makePSF
) {
    // Set up the shared mem cache
    const size_t cachesize {128};
    __shared__ char _cache[cachesize * (sizeof(ComplexLinearData<S>) + sizeof(S))];
    auto data_cache = reinterpret_cast<ComplexLinearData<S>*>(_cache);
    auto invlambdas_cache = reinterpret_cast<S*>(&data_cache[128]);

    // Get workunit information
    size_t rowstart, rowend, chanstart, chanend;
    double u0, v0, w0;
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
    const size_t nchanstep = std::min(nchans, cachesize);
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

            // Precompute phase in _meters_
            // We can convert to the dimensionless value later by a single
            // multiplication by the inverse lambda per channel
            auto [u, v, w] = uvws[irow];

            S theta = 2 * ::pi_v<S> * (u * l + v * m + w * n);  // [meters]
            S thetaoffset = 2 * ::pi_v<S> * (u0 * l + v0 * m + w0 * n);  // [dimensionless]

            for (size_t ichan {chanstart}; ichan < chanend; ichan += nchanstep) {
                const size_t N = min(nchanstep, chanend - ichan);

                // Populate cache
                __syncthreads();
                for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
                    // Copy global values to shared memory cache
                    data_cache[j] = data[irow * rowstride + ichan + j];
                    invlambdas_cache[j] =  1. / lambdas[ichan + j];

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
                    ComplexLinearData<S> datum = data_cache[j];
                    // datum = {1, 0, 0, 1};

                    auto phase = cis(theta * invlambdas_cache[j] - thetaoffset);
                    // auto invl = invlambdas_cache[j];
                    // S theta = 2 * ::pi_v<S> * (
                    //     (u * invl - u0) * l + (v * invl - v0) * m + (w * invl - w0) * n
                    // );
                    // auto phase = cis(theta);

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
        output = static_cast<T>(cell);

        // if constexpr(makePSF) {
        //     // No beam correction for PSF
        //     output = static_cast<T>(cell);
        // } else {
        //     // Grab A terms and apply beam corrections and normalization
        //     const auto Al = Aleft[idx].inv();
        //     const auto Ar = Aright[idx].inv().adjoint();

        //     // Apply beam to cell: inv(Aleft) * cell * inv(Aright)^H
        //     // Then conversion from LinearData to output T
        //     output = static_cast<T>(matmul(matmul(Al, cell), Ar));

        //     // Calculate norm
        //     T norm = T(matmul(Al, Ar).norm());

        //     // Finally, apply norm
        //     output /= norm;
        // }

        // Perform final FFT fft normalization and apply taper
        output *= subtaper[idx] / subgridspec.size();

        subgrids[blockIdx.y * subgridspec.size() + idx] = output;
    }
}

template <typename T, typename S>
void gridderk(
    DeviceSpan<T, 3> subgrids,
    const DeviceSpan<Visibility::Workunit, 1> workunits,
    const DeviceSpan<std::array<double, 3>, 1> uvws,
    const DeviceSpan<ComplexLinearData<S>, 2> data,
    const DeviceSpan<LinearData<S>, 2> weights,
    const GridSpec subgridspec,
    const DeviceSpan<double, 1> lambdas,
    const DeviceSpan<S, 2> subtaper,
    size_t rowoffset,
    bool makePSF
) {
    // x-dimension corresponds to cells in the subgrid
    int nthreadsx {128}; // hardcoded to match the cache size
    int nblocksx = cld<size_t>(subgridspec.size(), nthreadsx);

    // y-dimension corresponds to workunit index
    int nthreadsy {1};
    int nblocksy = workunits.size();

    auto fn = _gridderk<T, S>;
    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy, 1), dim3(nthreadsx, nthreadsy, 1),
        0, hipStreamPerThread,
        subgrids, workunits, uvws, data, weights, subgridspec, lambdas, subtaper, rowoffset, makePSF
    );
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void _addsubgridbatched(
    DeviceSpan<T, 2> grid, const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<Visibility::Workunit, 1> workunits, const DeviceSpan<T, 3> subgrids,
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
void addsubgridbatched(
    DeviceSpan<T, 2> grid, const DeviceSpan<size_t, 1> widxs,
    const DeviceSpan<Visibility::Workunit, 1> workunits, const DeviceSpan<T, 3> subgrids,
    const GridSpec gridspec, const GridSpec subgridspec
) {
    auto timer = Timer::get("invert::wlayer::gridder::thread::adder");

    auto fn = _addsubgridbatched<T>;
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

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invertbatch(
    Visibility& vis,
    std::vector<Visibility::Workunit>& workunits,
    GridConfig gridconf,
    bool makePSF = false
) {
    GridSpec gridspec = gridconf.padded();
    GridSpec subgridspec = gridconf.subgrid();

    DeviceArray<T<S>, 2> imgd(gridspec.Nx, gridspec.Ny);
    DeviceArray<T<S>, 2> wlayer(gridspec.Nx, gridspec.Ny);
    DeviceArray<S, 2> subtaperd {pswf<S>(subgridspec)};

    // TODO: set batch dynamically, based on memory allowance
    const size_t nbatch {2 * 8192};

    // TODO: Round up workunits to include any remaining channels

    hipfftHandle subplan {};
    int rank[] {(int) subgridspec.Ny, (int) subgridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &subplan, 2, rank,
        rank, 1, subgridspec.size(),
        rank, 1, subgridspec.size(),
        HIPFFT_C2C, nbatch
    ) );
    HIPFFTCHECK( hipfftSetStream(subplan, hipStreamPerThread) );

    // auto subplan = fftPlan<T<S>>(subgridspec);

    auto wplan = fftPlan<T<S>>(gridspec);

    for (size_t istart {}; istart < workunits.size(); istart += nbatch) {
        size_t iend = std::min(istart + nbatch, workunits.size());
        long long nworkunits = iend - istart;
        fmt::println("istart={}; iend={}; nworkunits={}; workunits.size={} {}", istart, iend, nworkunits, workunits.size(), istart < workunits.size());

        size_t rowstart = workunits[istart].rowstart;
        size_t rowend = workunits[iend - 1].rowend;
        long long nrows = rowend - rowstart;
        long long rowstride = vis.freqs.size();

        fmt::println("batching {} rows of data ({}-{}), with {} nworkunits", nrows, rowstart, rowend, nworkunits);

        HostSpan<Visibility::Workunit, 1> workunits_h(
            {nworkunits}, workunits.data() + istart
        );

        HostSpan<ComplexLinearData<S>, 2> data_h(
            {nrows, rowstride}, vis.data.data() + rowstart * rowstride
        );

        HostSpan<LinearData<S>, 2> weights_h(
            {nrows, rowstride}, vis.weights.data() + rowstart * rowstride
        );

        HostArray<std::array<double, 3>, 1> uvws_h({nrows});
        for (size_t i {}; i < nrows; ++i) {
            auto m = vis.metadata[rowstart + i];
            uvws_h[i] = {m.u, m.v, m.w};
        }

        // Copy across data
        DeviceArray<Visibility::Workunit, 1> workunits_d(workunits_h);
        DeviceArray<ComplexLinearData<S>, 2> data_d(data_h);
        DeviceArray<LinearData<S>, 2> weights_d(weights_h);
        DeviceArray<double, 1> lambdas_d(vis.lambdas);
        DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);

        // Allocate subgrid stack
        DeviceArray<T<S>, 3> subgrids_d(
            {subgridspec.Nx, subgridspec.Ny, nbatch}
        );

        // HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

        gridderk<T<S>, S>(
            subgrids_d, workunits_d, uvws_d, data_d, weights_d,
            subgridspec, lambdas_d, subtaperd, rowstart, makePSF
        );

        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        fmt::println("Calling FFT...");

        fftshift_batched(subgrids_d, FFTShift::pre);
        hipfftExecC2C(
            subplan,
            (hipfftComplex*) subgrids_d.data(),
            (hipfftComplex*) subgrids_d.data(),
            HIPFFT_FORWARD
        );
        fftshift_batched(subgrids_d, FFTShift::post);

        // Reset deltal, deltam shift prior to adding to master grid
        PIGI_TIMER(
            "invert::wlayer::gridder::thread::deltlm",
            map([
                =,
                deltal=static_cast<S>(subgridspec.deltal),
                deltam=static_cast<S>(subgridspec.deltam),
                stride=subgridspec.size()
            ] __device__ (auto idx, auto& px) {
                idx %= stride;
                auto [u, v] = subgridspec.linearToUV<S>(idx);
                px *= cispi(-2 * (u * deltal + v * deltam));
            }, Iota(), subgrids_d);
        );

        std::unordered_map<double, std::vector<size_t>> widxs;
        for (size_t i {istart}; i < iend; ++i) {
            auto workunit = workunits[i];
            widxs[workunit.w].push_back(i - istart);
        }

        fmt::println("Processing {} w-layers...", widxs.size());

        for (auto& [w0, idxs] : widxs) {
            wlayer.zero();

            addsubgridbatched(
                wlayer, DeviceArray<size_t, 1>(idxs),
                workunits_d, subgrids_d, gridspec, subgridspec
            );

            // Apply deltal, deltam shift
            map([
                =,
                deltal=static_cast<S>(gridspec.deltal),
                deltam=static_cast<S>(gridspec.deltam)
            ] __device__ (auto idx, auto& wlayer) {
                auto [u, v] = gridspec.linearToUV<S>(idx);
                wlayer *= cispi(2 * (u * deltal + v * deltam));
            }, Iota(), wlayer);

            // FFT the full wlayer
            fftExec(wplan, wlayer, HIPFFT_BACKWARD);

            // Apply wcorrection and append layer onto img
            map([gridspec=gridspec, w0=static_cast<S>(w0)] __device__ (auto idx, auto& imgd, auto wlayer) {
                auto [l, m] = gridspec.linearToSky<S>(idx);
                wlayer *= cispi(2 * w0 * ndash(l, m));
                imgd += wlayer;
            }, Iota(), imgd, wlayer);
        }

        {
            size_t free, total;
            HIPCHECK( hipMemGetInfo(&free, &total) );
            fmt::println(
                "Free: {} GB Total: {} GB ",
                free / 1024. / 1024. / 1024., total / 1024. / 1024. / 1024.
            );
        }
    }

    // Copy image from device to host
    HostArray<T<S>, 2> img(imgd);

    hipfftDestroy(subplan);
    hipfftDestroy(wplan);

    HIPCHECK(hipStreamSynchronize(hipStreamPerThread));

    // The final image still has a taper applied. It's time to remove it.
    img /= pswf<S>(gridspec);

    return resize(img, gridconf.padded(), gridconf.grid());
}