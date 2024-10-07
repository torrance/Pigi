#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"

template <typename T, int N>
__global__
void _fftshift(
    DeviceSpan<T, N> grid, GridSpec gridspec, long long lpx0, long long mpx0, size_t nbatches
) requires (N == 2 || N == 3) {
    for (
        size_t batchid {blockIdx.y}; batchid < nbatches; batchid += blockDim.y * gridDim.y
    ) {
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

    // In the batched case, each iteration along the third axis is an independent grid
    size_t nbatch = N == 2 ? 1 : grid.size(2);

    auto fn = _fftshift<T, N>;

    auto [nblocksx, nthreadsx] = getKernelConfig(fn, gridspec.size());

    uint32_t nthreadsy {1};
    uint32_t nblocksy = std::min<size_t>(nbatch, 65535);

    hipLaunchKernelGGL(
        fn, dim3(nblocksx, nblocksy), dim3(nthreadsx, nthreadsy), 0, hipStreamPerThread,
        grid, gridspec, lpx0, mpx0, nbatch
    );
}

template <typename T>
hipfftHandle fftPlan(const GridSpec gridspec, long long nbatch=1) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(static_cast<int>(sizeof(T)) == -1, "No fftPlan specialisation provided");
    [[maybe_unused]] GridSpec g = gridspec;
    [[maybe_unused]] long long n = nbatch;
    hipfftHandle plan;
    return plan;
}

template<>
hipfftHandle fftPlan<thrust::complex<float>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 1, gridspec.size(), HIP_C_32F,
        rank, 1, gridspec.size(), HIP_C_32F,
        nbatch, &worksize, HIP_C_32F
    ) );

    return plan;
}

template<>
hipfftHandle fftPlan<thrust::complex<double>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 1, gridspec.size(), HIP_C_64F,
        rank, 1, gridspec.size(), HIP_C_64F,
        nbatch, &worksize, HIP_C_64F
    ) );

    return plan;
}

template <>
hipfftHandle fftPlan<ComplexLinearData<float>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 4, gridspec.size() * 4, HIP_C_32F,
        rank, 4, gridspec.size() * 4, HIP_C_32F,
        nbatch, &worksize, HIP_C_32F
    ) );

    return plan;
}

template<>
hipfftHandle fftPlan<ComplexLinearData<double>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR

    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 4, gridspec.size() * 4, HIP_C_64F,
        rank, 4, gridspec.size() * 4, HIP_C_64F,
        nbatch, &worksize, HIP_C_64F
    ) );

    return plan;
}

template<>
hipfftHandle fftPlan<StokesI<float>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 1, gridspec.size(), HIP_C_32F,
        rank, 1, gridspec.size(), HIP_C_32F,
        nbatch, &worksize, HIP_C_32F
    ) );

    return plan;
}

template<>
hipfftHandle fftPlan<StokesI<double>>(const GridSpec gridspec, long long nbatch) {
    hipfftHandle plan {};
    HIPFFTCHECK( hipfftCreate(&plan) );
    HIPFFTCHECK( hipfftSetAutoAllocation(plan, true) );
    HIPFFTCHECK( hipfftSetStream(plan, hipStreamPerThread) );

    size_t worksize;
    long long rank[] {gridspec.Ny, gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftXtMakePlanMany(
        plan, 2, rank,
        rank, 1, gridspec.size(), HIP_C_64F,
        rank, 1, gridspec.size(), HIP_C_64F,
        nbatch, &worksize, HIP_C_64F
    ) );

    return plan;
}

template <typename T>
size_t fftEstimate(const GridSpec gridspec, long long nbatch=1) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(static_cast<int>(sizeof(T)) == -1, "No fftEstimate specialisation provided");
    [[maybe_unused]] GridSpec g = gridspec;
    [[maybe_unused]] int n = nbatch;
    return {};
}

template<>
size_t fftEstimate<thrust::complex<float>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_C2C, nbatch, &worksize
    ) );

    return worksize;
}

template<>
size_t fftEstimate<thrust::complex<double>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_Z2Z, nbatch, &worksize
    ) );

    return worksize;
}

template<>
size_t fftEstimate<ComplexLinearData<float>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 4, gridspec.size() * 4,
        rank, 4, gridspec.size() * 4,
        HIPFFT_C2C, nbatch, &worksize
    ) );

    return worksize;
}

template<>
size_t fftEstimate<ComplexLinearData<double>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 4, gridspec.size() * 4,
        rank, 4, gridspec.size() * 4,
        HIPFFT_Z2Z, nbatch, &worksize
    ) );

    return worksize;
}

template<>
size_t fftEstimate<StokesI<float>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_C2C, nbatch, &worksize
    ) );

    return worksize;
}

template<>
size_t fftEstimate<StokesI<double>>(const GridSpec gridspec, long long nbatch) {
    size_t worksize {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftEstimateMany(
        2, rank,
        rank, 1, gridspec.size(),
        rank, 1, gridspec.size(),
        HIPFFT_Z2Z, nbatch, &worksize
    ) );

    return worksize;
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<thrust::complex<float>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    HIPFFTCHECK( hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    ) );
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<thrust::complex<double>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    HIPFFTCHECK( hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    ) );
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<float>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    for (long long i {}; i < 4; ++i) {
        HIPFFTCHECK( hipfftExecC2C(
            plan,
            (hipfftComplex*) grid.data() + i,
            (hipfftComplex*) grid.data() + i,
            direction
        ) );
    }
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<double>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    for (long long i {}; i < 4; ++i) {
        HIPFFTCHECK( hipfftExecZ2Z(
            plan,
            (hipfftDoubleComplex*) grid.data() + i,
            (hipfftDoubleComplex*) grid.data() + i,
            direction
        ) );
    }
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<float>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    HIPFFTCHECK( hipfftExecC2C(
        plan,
        (hipfftComplex*) grid.data(),
        (hipfftComplex*) grid.data(),
        direction
    ) );
    fftshift(grid, FFTShift::post);
}

template <int N>
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<double>, N> grid, int direction) {
    fftshift(grid, FFTShift::pre);
    HIPFFTCHECK( hipfftExecZ2Z(
        plan,
        (hipfftDoubleComplex*) grid.data(),
        (hipfftDoubleComplex*) grid.data(),
        direction
    ) );
    fftshift(grid, FFTShift::post);
}