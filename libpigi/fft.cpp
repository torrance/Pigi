#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "array.cpp"
#include "hip.cpp"
#include "gridspec.cpp"
#include "outputtypes.cpp"

template <typename T>
__global__
void _fftshift(SpanMatrix<T> grid, GridSpec gridspec) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.Nx * gridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        auto [lpx, mpx] = gridspec.linearToGrid(idx);

        auto lfactor {1 - 2 * (lpx % 2)};
        auto mfactor {1 - 2 * (mpx % 2)};

        grid[idx] *= (lfactor * mfactor);
    }
}

template<typename T>
void fftshift(SpanMatrix<T> grid) {
    // Create dummy GridSpec so that we have access to linearToGrid() method
    GridSpec gridspec {
        static_cast<long long>(grid.size(0)),
        static_cast<long long>(grid.size(1)),
        0, 0
    };

    auto [nblocks, nthreads] = getKernelConfig(
        _fftshift<T>, gridspec.Nx * gridspec.Ny
    );

    hipLaunchKernelGGL(
        _fftshift<T>, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec
    );
}

template <typename T>
hipfftHandle fftPlan([[maybe_unused]] GridSpec gridspec) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(sizeof(T) == -1, "No fftPlan specialisation provided");
}

template <>
hipfftHandle fftPlan<ComplexLinearData<float>>(GridSpec gridspec) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 4, 1,
        rank, 4, 1,
        HIPFFT_C2C, 4
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    return plan;
}

template<>
hipfftHandle fftPlan<ComplexLinearData<double>>(GridSpec gridspec) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 4, 1,
        rank, 4, 1,
        HIPFFT_Z2Z, 4
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    return plan;
}

template<>
hipfftHandle fftPlan<StokesI<float>>(GridSpec gridspec) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 1, 1,
        rank, 1, 1,
        HIPFFT_C2C, 1
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    return plan;
}

template<>
hipfftHandle fftPlan<StokesI<double>>(GridSpec gridspec) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 1, 1,
        rank, 1, 1,
        HIPFFT_Z2Z, 1
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    return plan;
}

auto fftExec(hipfftHandle plan, SpanMatrix<ComplexLinearData<float>> grid, int direction) {
    fftshift(grid);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}

auto fftExec(hipfftHandle plan, SpanMatrix<ComplexLinearData<double>> grid, int direction) {
    fftshift(grid);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}

auto fftExec(hipfftHandle plan, SpanMatrix<StokesI<float>> grid, int direction) {
    fftshift(grid);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}
auto fftExec(hipfftHandle plan, SpanMatrix<StokesI<double>> grid, int direction) {
    fftshift(grid);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}