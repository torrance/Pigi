#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"

template <typename T>
__global__
void _fftshift(DeviceSpan<T, 2> grid, GridSpec gridspec, long long lpx0, long long mpx0) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        // Get grid pixel values and offset to origin
        auto [lpx, mpx] = gridspec.linearToGrid(idx);
        lpx -= lpx0;
        mpx -= mpx0;

        auto factor {1 - 2 * ((lpx + mpx) & 1)};
        grid[idx] *= factor;
    }
}

enum class FFTShift { pre, post };

template<typename T>
void fftshift(DeviceSpan<T, 2> grid, FFTShift stage) {
    // Create dummy GridSpec so that we have access to linearToGrid() method
    GridSpec gridspec {.Nx=grid.size(0), .Ny=grid.size(1)};

    // In Fourier domain where the power is centered, the checkerboard pattern must
    // be centered with +1 on the central pixel. However, in the image domain, this
    // the checkerboard pattern is simply with respect to the 0th pixel. (In practice, it
    // doesn't matter which domain is which, as long as one of them is offset with respect
    // to the central pixel).
    long long lpx0 {}, mpx0 {};
    if (stage == FFTShift::pre) {
        lpx0 = gridspec.Nx / 2;
        mpx0 = gridspec.Ny / 2;
    }

    auto [nblocks, nthreads] = getKernelConfig(
        _fftshift<T>, gridspec.size()
    );

    hipLaunchKernelGGL(
        _fftshift<T>, nblocks, nthreads, 0, hipStreamPerThread,
        grid, gridspec, lpx0, mpx0
    );
}

template <typename T>
__global__
void _fftshift_batched(DeviceSpan<T, 3> grid, GridSpec gridspec, long long lpx0, long long mpx0) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        // Get grid pixel values and offset to origin
        auto [lpx, mpx] = gridspec.linearToGrid(idx);
        lpx -= lpx0;
        mpx -= mpx0;

        auto factor {1 - 2 * ((lpx + mpx) & 1)};
        grid[gridspec.size() * blockIdx.y + idx] *= factor;
    }
}

template<typename T>
void fftshift_batched(DeviceSpan<T, 3> grid, FFTShift stage) {
    // Create dummy GridSpec so that we have access to linearToGrid() method
    GridSpec gridspec {.Nx=grid.size(0), .Ny=grid.size(1)};

    // In Fourier domain where the power is centered, the checkerboard pattern must
    // be centered with +1 on the central pixel. However, in the image domain, this
    // the checkerboard pattern is simply with respect to the 0th pixel. (In practice, it
    // doesn't matter which domain is which, as long as one of them is offset with respect
    // to the central pixel).
    long long lpx0 {}, mpx0 {};
    if (stage == FFTShift::pre) {
        lpx0 = gridspec.Nx / 2;
        mpx0 = gridspec.Ny / 2;
    }

    auto [nblocksx, nthreadsx] = getKernelConfig(
        _fftshift_batched<T>, gridspec.size()
    );

    int nthreadsy {1};
    int nblocksy = grid.size(2);

    hipLaunchKernelGGL(
        _fftshift_batched<T>, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy),
        0, hipStreamPerThread, grid, gridspec, lpx0, mpx0
    );
}

template <typename T>
hipfftHandle fftPlan([[maybe_unused]] const GridSpec gridspec) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(static_cast<int>(sizeof(T)) == -1, "No fftPlan specialisation provided");
    hipfftHandle plan;
    return plan;
}

template<>
hipfftHandle fftPlan<thrust::complex<float>>(const GridSpec gridspec) {
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
hipfftHandle fftPlan<thrust::complex<double>>(const GridSpec gridspec) {
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

void fftExec(hipfftHandle plan, DeviceSpan<thrust::complex<float>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

void fftExec(hipfftHandle plan, DeviceSpan<thrust::complex<double>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<float>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<double>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

void fftExec(hipfftHandle plan, DeviceSpan<StokesI<float>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

void fftExec(hipfftHandle plan, DeviceSpan<StokesI<double>, 2> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}