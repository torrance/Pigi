#pragma once

#include <hipfft/hipfft.h>

#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"

template <typename T>
hipfftHandle fftPlan([[maybe_unused]] const GridSpec gridspec) {
    // This is a dummy template that allows the following specialisations.
    // It should never be instantiated, only the specialisations are allowed.
    static_assert(sizeof(T) == -1, "No fftPlan specialisation provided");
    hipfftHandle plan;
    return plan;
}
template <> hipfftHandle fftPlan<std::complex<float>>(const GridSpec gridspec);
template <> hipfftHandle fftPlan<std::complex<double>>(const GridSpec gridspec);
template <> hipfftHandle fftPlan<ComplexLinearData<float>>(const GridSpec gridspec);
template <> hipfftHandle fftPlan<ComplexLinearData<double>>(const GridSpec gridspec);
template <> hipfftHandle fftPlan<StokesI<float>>(const GridSpec gridspec);
template <> hipfftHandle fftPlan<StokesI<double>>(const GridSpec gridspec);

void fftExec(hipfftHandle plan, DeviceSpan<std::complex<float>, 2> grid, int direction);
void fftExec(hipfftHandle plan, DeviceSpan<std::complex<double>, 2> grid, int direction);
void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<float>, 2> grid, int direction);
void fftExec(hipfftHandle plan, DeviceSpan<ComplexLinearData<double>, 2> grid, int direction);
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<float>, 2> grid, int direction);
void fftExec(hipfftHandle plan, DeviceSpan<StokesI<double>, 2> grid, int direction);
