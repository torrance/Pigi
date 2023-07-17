#pragma once

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

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
        auto [lpx, mpx] = linearToXY(idx, gridspec);

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

auto fftExec(hipfftHandle plan, LinearData<double>* grid, int direction) { 
    return hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
}
auto fftExec(hipfftHandle plan, StokesI<float>* grid, int direction) { 
    return hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid,
        (hipfftComplex*) grid,
        direction
    );
}
auto fftExec(hipfftHandle plan, StokesI<double>* grid, int direction) { 
    return hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
}

auto fftType(LinearData<double>* grid) { return HIPFFT_Z2Z; }
auto fftType(StokesI<float>* grid) { return HIPFFT_C2C; }
auto fftType(StokesI<double>* grid) { return HIPFFT_Z2Z; }