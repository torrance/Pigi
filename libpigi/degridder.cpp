#pragma once

#include <algorithm>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "array.cpp"
#include "fft.cpp"
#include "hip.cpp"
#include "util.cpp"
#include "uvdatum.cpp"
#include "workunit.cpp"

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
    SpanVector<UVDatum<T>> uvdata,
    const UVWOrigin<T> origin,
    const SpanMatrix<ComplexLinearData<T>> subgrid,
    const GridSpec subgridspec,
    const DegridOp degridop
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < uvdata.size();
        idx += blockDim.x * gridDim.x
    ) {
        UVDatum<T> uvdatum {uvdata[idx]};

        ComplexLinearData<T> data;
        for (int i {}; i < subgridspec.size(); ++i) {
            auto [l, m] = subgridspec.linearToSky<T>(i);
            auto phase = cispi(-2 * (
                (uvdatum.u - origin.u0) * l +
                (uvdatum.v - origin.v0) * m +
                (uvdatum.w - origin.w0) * ndash(l, m)
            ));
            auto cell = subgrid[i];
            cell *= phase;
            data += cell;
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

        uvdata[idx] = uvdatum;
    }
}

template <typename T>
void gpudft(
    SpanVector<UVDatum<T>> uvdata,
    const UVWOrigin<T> origin,
    const SpanMatrix<ComplexLinearData<T>> subgrid,
    const GridSpec subgridspec,
    const DegridOp degridop
) {
    auto fn = _gpudft<T>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, uvdata.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        uvdata, origin, subgrid, subgridspec, degridop
    );
}

template <typename T, typename S>
__global__
void _extractSubgrid(
    SpanMatrix<ComplexLinearData<S>> subgrid, const GridSpec subgridspec,
    const SpanMatrix<T> grid, const GridSpec gridspec,
    const long long u0px, const long long v0px
) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgridspec.size();
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
            // This assignment performs an implicit conversion
            subgrid[idx] = grid[gridspec.gridToLinear(upx, vpx)];
        }
    }
}

template <typename T, typename S>
auto extractSubgrid(
    const SpanMatrix<T> grid,
    const WorkUnit<S>& workunit
) {
    // Allocate subgrid matrix
    DeviceMatrix<ComplexLinearData<S>> subgrid(
        {workunit.subgridspec.Nx, workunit.subgridspec.Ny}
    );

    // Create dummy gridspec to have access to gridToLinear() method
    GridSpec gridspec {
        static_cast<long long>(grid.size(0)),
        static_cast<long long>(grid.size(1)),
        0, 0
    };

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

template <typename T>
__global__
void _applyAterms(
    SpanMatrix<ComplexLinearData<T>> subgrid,
    const SpanMatrix<ComplexLinearData<T>> Aleft,
    const SpanMatrix<ComplexLinearData<T>> Aright,
    const SpanMatrix<T> subtaper,
    const size_t norm
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgrid.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto cell = subgrid[idx];
        auto Al = Aleft[idx];
        auto Ar = Aright[idx];
        auto t = subtaper[idx];

        cell.lmul(Al).rmul(Ar.adjoint());
        cell *= (t / norm);
        subgrid[idx] = cell;
    }
}

template <typename T>
void applyAterms(
    SpanMatrix<ComplexLinearData<T>> subgrid,
    const SpanMatrix<ComplexLinearData<T>> Aleft,
    const SpanMatrix<ComplexLinearData<T>> Aright,
    const SpanMatrix<T> subtaper
) {
    auto fn = _applyAterms<T>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, subgrid.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        subgrid, Aleft, Aright, subtaper, subgrid.size()
    );
}

template <typename T, typename S>
void degridder(
    SpanVector<WorkUnit<S>> workunits,
    const SpanMatrix<T> grid,
    const SpanMatrix<S> subtaper,
    const DegridOp degridop
) {
    // Transfer Aterms to GPU, since these are often shared
    std::unordered_map<
        const ComplexLinearData<S>*, const DeviceMatrix<ComplexLinearData<S>>
    > Aterms;
    for (const auto& workunit : workunits) {
        Aterms.try_emplace(workunit.Aleft.data(), workunit.Aleft);
        Aterms.try_emplace(workunit.Aright.data(), workunit.Aright);
    }

    auto plan = fftPlan<ComplexLinearData<S>>(workunits.front().subgridspec);

    for (auto& workunit : workunits) {
        // Transfer data to device and retrieve A terms
        DeviceVector<UVDatum<S>> uvdata {workunit.data};
        const auto& Aleft = Aterms.at(workunit.Aleft.data());
        const auto& Aright = Aterms.at(workunit.Aright.data());

        // Allocate subgrid and extract from grid
        auto subgrid = extractSubgrid(grid, workunit);

        fftExec(plan, subgrid, HIPFFT_BACKWARD);

        // Apply aterms, taper and normalize post-FFT
        applyAterms<S>(subgrid, Aleft, Aright, subtaper);

        gpudft<S>(
            uvdata,
            {workunit.u0, workunit.v0, workunit.w0},
            subgrid, workunit.subgridspec, degridop
        );

        // Transfer data back to host
        // TODO: Encapsulate this explicit data transfer somehow
        HIPCHECK( hipMemcpyDtoHAsync(
            workunit.data.data(), uvdata.data(),
            uvdata.size() * sizeof(UVDatum<S>), hipStreamPerThread
        ) );

        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
    }
}