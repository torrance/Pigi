#include "fft.h"

#include <hip/hip_runtime.h>

#include "hip.h"

template <typename T>
__global__
void _fftshift(DeviceSpan<T, 2> grid, GridSpec gridspec) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [lpx, mpx] = gridspec.linearToGrid(idx);

        auto lfactor {1 - 2 * (lpx % 2)};
        auto mfactor {1 - 2 * (mpx % 2)};

        grid[idx] *= (lfactor * mfactor);
    }
}

template<typename T>
void fftshift(DeviceSpan<T, 2> grid) {
    // Create dummy GridSpec so that we have access to linearToGrid() method
    GridSpec gridspec {grid.size(0), grid.size(1), 0, 0};

    auto [nblocks, nthreads] = getKernelConfig(
        _fftshift<T>, gridspec.Nx * gridspec.Ny
    );

    hipLaunchKernelGGL(
        _fftshift<T>, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec
    );
}

template <>
hipfftHandle fftPlan<ComplexLinearData<float>>(const GridSpec gridspec) {
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
hipfftHandle fftPlan<ComplexLinearData<double>>(const GridSpec gridspec) {
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
hipfftHandle fftPlan<StokesI<float>>(const GridSpec gridspec) {
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
hipfftHandle fftPlan<StokesI<double>>(const GridSpec gridspec) {
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

void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<float>, 2> grid, int direction) {
    fftshift(grid);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}

void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<double>, 2> grid, int direction) {
    fftshift(grid);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}

void fftExec(hipfftHandle plan, DeviceSpan<StokesI<float>, 2> grid, int direction) {
    fftshift(grid);
    hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}

void fftExec(hipfftHandle plan, DeviceSpan<StokesI<double>, 2> grid, int direction) {
    fftshift(grid);
    hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid);
}