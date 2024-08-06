#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"

template <typename T, int N>
__global__
void _fftshift(
    DeviceSpan<T, N> grid, GridSpec gridspec, long long lpx0, long long mpx0
) requires (N == 2 || N == 3) {
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

enum class FFTShift { pre, post };

template<typename T, int N>
void fftshift(DeviceSpan<T, N> grid, FFTShift stage) requires (N == 2 || N == 3) {
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

    auto fn = _fftshift<T, N>;

    auto [nblocksx, nthreadsx] = getKernelConfig(fn, gridspec.size());

    // In the batched case, each iteration along the third axis is an independent grid
    int nthreadsy {1};
    int nblocksy = N == 2 ? 1 : grid.size(2);

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        grid, gridspec, lpx0, mpx0
    );
}

template <typename T>
hipfftHandle fftPlan([[maybe_unused]] const GridSpec gridspec, int nbatch=1) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(static_cast<int>(sizeof(T)) == -1, "No fftPlan specialisation provided");
    hipfftHandle plan;
    return plan;
}

template<>
hipfftHandle fftPlan<thrust::complex<float>>(const GridSpec gridspec, int) {
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
hipfftHandle fftPlan<thrust::complex<double>>(const GridSpec gridspec, int) {
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
hipfftHandle fftPlan<ComplexLinearData<float>>(const GridSpec gridspec, int) {
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
hipfftHandle fftPlan<ComplexLinearData<double>>(const GridSpec gridspec, int) {
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
hipfftHandle fftPlan<StokesI<float>>(const GridSpec gridspec, int nbatch) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_C2C, nbatch
    ) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    return plan;
}

template<>
hipfftHandle fftPlan<StokesI<double>>(const GridSpec gridspec, int nbatch) {
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_Z2Z, nbatch
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

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<float>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<double>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    );
    fftshift(grid, FFTShift::post);
}