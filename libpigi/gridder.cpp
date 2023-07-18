#pragma once

#include <complex>

#include <hip/hip_runtime.h>

#include "array.cpp"
#include "gridspec.cpp"
#include "outputtypes.cpp"
#include "util.cpp"
#include "uvdatum.cpp"
#include "workunit.cpp"

template <typename T, typename S>
__global__
void gpudift(
    T * __restrict__ subgrid,
    const ComplexLinearData<S> * __restrict__ Aleft,
    const ComplexLinearData<S> * __restrict__ Aright,
    const UVWOrigin<S> origin,
    const UVDatum<S> * __restrict__ uvdata,
    const size_t uvdata_n,
    const GridSpec subgridspec
) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgridspec.Nx * subgridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        auto [l, m] = subgridspec.linearToSky<S>(idx);
        S n {ndash(l, m)};

        ComplexLinearData<S> cell {};
        for (size_t i = 0; i < uvdata_n; ++i) {
            auto uvdatum = uvdata[i];
            auto phase = cispi(
                2 * ((uvdatum.u - origin.u0) * l + (uvdatum.v - origin.v0) * m + (uvdatum.w - origin.w0) * n)
            );

            uvdatum.data *= uvdatum.weights;
            uvdatum.data *= phase;
            cell += uvdatum.data;
        }

        // TODO: add beam correction and normalize
        subgrid[idx] = cell;
    }
}

template <typename T, typename S>
__global__
void applytaper(
    T* __restrict__ subgrid, S* __restrict__ taper, GridSpec subgridspec
) {
    auto N = subgridspec.Nx * subgridspec.Ny;
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        subgrid[idx] *= (taper[idx] / N);
    }
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void addsubgrid(
    T* __restrict__ grid, GridSpec gridspec,
    T* __restrict__ subgrid, GridSpec subgridspec,
    long long u0px, long long v0px
) {
    // Iterate over each element of the subgrid
    auto N = subgridspec.Nx * subgridspec.Ny;
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        auto [upx, vpx] = subgridspec.linearToGrid(idx);

        // Transform to pixel position wrt to master grid
        upx += u0px - subgridspec.Nx / 2;
        vpx += v0px - subgridspec.Ny / 2;

        if (
            0 <= upx && upx < gridspec.Nx &&
            0 <= vpx && vpx < gridspec.Ny
        ) {
            grid[gridspec.gridToLinear(upx, vpx)] += subgrid[idx];
        }
    }
}

template<typename T, typename S>
void gridder(
    SpanMatrix<T> grid,
    const SpanVector<WorkUnit<S>> workunits,
    const SpanMatrix<S> subtaper
) {
    auto subgridspec = workunits[0].subgridspec;
    DeviceArray<T, 2> subgrid({(size_t) subgridspec.Nx, (size_t) subgridspec.Ny});

    // Make FFT plan
    auto plan = fftPlan(subgridspec, subgrid.data());

    for (const WorkUnit<S>& workunit : workunits) {
        UVWOrigin origin {workunit.u0, workunit.v0, workunit.w0};

        auto uvdata = workunit.data;
        auto Aleft = workunit.Aleft;
        auto Aright = workunit.Aright;

        // DFT
        {
            auto fn = gpudift<T, S>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx * subgridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, hipStreamPerThread,
                subgrid.data(), Aleft.data(),
                Aright.data(), origin, uvdata.data(),
                uvdata.size(), subgridspec
            );
        }

        // Taper
        {
            auto fn = applytaper<T, S>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx * subgridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, hipStreamPerThread,
                subgrid.data(), subtaper.data(), subgridspec
            );
        }

        // FFT
        fftExec(plan, subgrid.data(), subgridspec, HIPFFT_FORWARD);

        // Add back to master grid
        {
            auto fn = addsubgrid<T>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx, subgridspec.Ny
            );
            GridSpec gridspec {(long long) grid.size(0), (long long) grid.size(1), 0, 0};
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, hipStreamPerThread,
                grid.data(), gridspec, subgrid.data(), subgridspec, workunit.u0px, workunit.v0px
            );
        }
    }

    HIPFFTCHECK( hipfftDestroy(plan) );
}