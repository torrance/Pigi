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
void _gpudift(
    SpanMatrix<T> subgrid,
    const SpanMatrix< ComplexLinearData<S> > Aleft,
    const SpanMatrix< ComplexLinearData<S> > Aright,
    const UVWOrigin<S> origin,
    const SpanVector< UVDatum<S> > uvdata,
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
        for (auto uvdatum : uvdata) {
            auto phase = cispi(
                2 * ((uvdatum.u - origin.u0) * l + (uvdatum.v - origin.v0) * m + (uvdatum.w - origin.w0) * n)
            );

            uvdatum.data *= uvdatum.weights;
            uvdatum.data *= phase;
            cell += uvdatum.data;
        }

        subgrid[idx] = T::fromBeam(cell, Aleft[idx], Aright[idx]);
    }
}

template <typename T, typename S>
void gpudift(
    SpanMatrix<T> subgrid,
    const SpanMatrix< ComplexLinearData<S> > Aleft,
    const SpanMatrix< ComplexLinearData<S> > Aright,
    const UVWOrigin<S> origin,
    const SpanVector< UVDatum<S> > uvdata,
    const GridSpec subgridspec
) {
    auto fn = gpudift<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgridspec.Nx * subgridspec.Ny
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        subgrid, Aleft, Aright,
        origin, uvdata, subgridspec
    );
}

template <typename T, typename S>
__global__
void _applytaper(
    SpanMatrix<T> grid, const SpanMatrix<S> taper, const GridSpec gridspec
) {
    auto N = gridspec.Nx * gridspec.Ny;
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        grid[idx] *= (taper[idx] / N);
    }
}

template <typename T, typename S>
void applytaper(
    SpanMatrix<T> grid, const SpanMatrix<S> taper, const GridSpec gridspec
) {
    auto fn = applytaper<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, gridspec.Nx * gridspec.Ny
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        grid, taper, gridspec
    );
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void _addsubgrid(
    SpanMatrix<T> grid, const GridSpec gridspec,
    const SpanMatrix<T> subgrid, const GridSpec subgridspec,
    const long long u0px, const long long v0px
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

template <typename T>
void addsubgrid(
    SpanMatrix<T> grid,
    const SpanMatrix<T> subgrid, const GridSpec subgridspec,
    const long long u0px, const long long v0px
) {
    // Create dummy gridspec to have access to gridToLinear() method
    GridSpec gridspec {
        static_cast<long long>(grid.size(0)),
        static_cast<long long>(grid.size(1)),
        0, 0
    };

    auto fn = _addsubgrid<T>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgridspec.Nx, subgridspec.Ny
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec, subgrid, subgridspec, u0px, v0px
    );
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
    auto plan = fftPlan(subgrid);

    for (const WorkUnit<S>& workunit : workunits) {
        UVWOrigin origin {workunit.u0, workunit.v0, workunit.w0};

        auto uvdata = workunit.data;
        auto Aleft = workunit.Aleft;
        auto Aright = workunit.Aright;

        // DFT
        gpudift<T, S>(
            subgrid, Aleft, Aright, origin, uvdata, subgridspec
        );

        // Taper
        applytaper<T, S>(subgrid, subtaper, subgridspec);

        // FFT
        fftExec(plan, subgrid, HIPFFT_FORWARD);

        // Add back to master grid
        addsubgrid<T>(grid, subgrid, subgridspec, workunit.u0px, workunit.v0px);
    }

    HIPFFTCHECK( hipfftDestroy(plan) );
}