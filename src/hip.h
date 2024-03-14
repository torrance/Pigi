#pragma once

#include <fmt/format.h>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#define HIPCHECK(res) { hipcheck(res, __FILE__, __LINE__); }

inline void hipcheck(hipError_t res, const char* file, int line) {
    if (res != hipSuccess) {
        throw std::runtime_error(fmt::format(
            "Fatal hipError: {} (code {}) on line {} of {}",
            hipGetErrorString(res), static_cast<int>(res), line, file
        ));
    }
}

#define HIPFFTCHECK(res) { hipfftcheck(res, __FILE__, __LINE__); }

inline void hipfftcheck(hipfftResult res, const char* file, int line) {
    if (res != HIPFFT_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "Fatal hipfftError: {} file: {}, line: {}",
            (int) res, file, line
        ));
    }
}

template <typename T>
auto getKernelConfig(T fn, int N, size_t sharedMem=0) {
    static int nblocksmax, nthreads;

    [[maybe_unused]] static auto _ = [&]() {
        HIPCHECK( hipOccupancyMaxPotentialBlockSize(
            &nblocksmax, &nthreads, fn, sharedMem, 0
        ) );
        return true;
    }();

    return std::make_tuple(
        std::min<int>(nblocksmax, N / nthreads + 1), nthreads
    );
};