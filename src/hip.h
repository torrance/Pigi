#pragma once

#include <fmt/format.h>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#define HIPCHECK(res) { hipcheck(res, __FILE__, __LINE__); }

inline void hipcheck(hipError_t res, const char* file, int line) {
    if (res != hipSuccess) {
        fmt::println(
            "Fatal hipError: {}, file: {}, line: {}",
            hipGetErrorString(res), file, line
        );
        abort();
    }
}

#define HIPFFTCHECK(res) { hipfftcheck(res, __FILE__, __LINE__); }

inline void hipfftcheck(hipfftResult res, const char* file, int line) {
    if (res != HIPFFT_SUCCESS) {
        fmt::println(
            "Fatal hipfftError: {} file: {}, line: {}",
            (int) res, file, line
        );
        abort();
    }
}

template <typename T>
auto getKernelConfig(T fn, int N, size_t sharedMem=0) {
    static int nblocksmax, nthreads;

    [[maybe_unused]] static auto _ = [&]() {
        fmt::println("Calculating kernel configuration...");
        HIPCHECK( hipOccupancyMaxPotentialBlockSize(
            &nblocksmax, &nthreads, fn, sharedMem, 0
        ) );
        fmt::println("Recommended launch config: blocksmax={}, threads={}", nblocksmax, nthreads);
        return true;
    }();

    return std::make_tuple(
        std::min<int>(nblocksmax, N / nthreads + 1), nthreads
    );
};

template <typename F, typename T, typename... Ts> __global__
void _map(size_t N, F f, T x, Ts... xs);

#ifdef __HIPCC__
template <typename F, typename T, typename... Ts>
__global__
void _map(size_t N, F f, T x, Ts... xs) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        f(x[idx], xs[idx]...);
    }
}
#endif

template<typename F, typename T, typename... Ts>
void map(F f, T x, Ts... xs) {
    size_t N { std::min({x.size(), xs.size()...}) };
    auto fn = _map<F, T, Ts...>;
    auto [nblocks, nthreads] = getKernelConfig(fn, N);
    hipLaunchKernelGGL(fn, nblocks, nthreads, 0, hipStreamPerThread, N, f, x, xs...);
}
