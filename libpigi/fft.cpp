#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "hip.cpp"
#include "gridspec.cpp"
#include "outputtypes.cpp"

template <typename T>
__global__
void applyCheckerboard(T* grid, GridSpec gridspec) {
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
void fftshift(T* grid, GridSpec gridspec) {
    auto [nblocks, nthreads] = getKernelConfig(
        applyCheckerboard<T>, gridspec.Nx * gridspec.Ny
    );

    hipLaunchKernelGGL(
        applyCheckerboard<T>, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec
    );
}

hipfftHandle fftPlan(GridSpec gridspec, [[maybe_unused]] ComplexLinearData<float>* _) {
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

hipfftHandle fftPlan(GridSpec gridspec, [[maybe_unused]] ComplexLinearData<double>* _) {
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

hipfftHandle fftPlan(GridSpec gridspec, [[maybe_unused]] StokesI<float>* _) {
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

hipfftHandle fftPlan(GridSpec gridspec, [[maybe_unused]] StokesI<double>* _) {
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

auto fftExec(hipfftHandle plan, ComplexLinearData<float>* grid, GridSpec gridspec, int direction) { 
    fftshift(grid, gridspec);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid,
        (hipfftComplex*) grid,
        direction
    );
    fftshift(grid, gridspec);
}

auto fftExec(hipfftHandle plan, ComplexLinearData<double>* grid, GridSpec gridspec, int direction) { 
    fftshift(grid, gridspec);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
    fftshift(grid, gridspec);
}

auto fftExec(hipfftHandle plan, StokesI<float>* grid, GridSpec gridspec, int direction) {
    fftshift(grid, gridspec);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid,
        (hipfftComplex*) grid,
        direction
    );
    fftshift(grid, gridspec);
}
auto fftExec(hipfftHandle plan, StokesI<double>* grid, GridSpec gridspec, int direction) {
    fftshift(grid, gridspec);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
    fftshift(grid, gridspec);
}