#pragma once

#include <mutex>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <fmt/format.h>

#include "logger.h"

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

class GPU {
public:
    GPU(const GPU&) = delete;
    GPU(GPU&&) = delete;
    GPU& operator=(const GPU&) = delete;
    GPU& operator=(GPU&&) = delete;

    static GPU& getInstance() {
        static GPU instance;
        return instance;
    }

    int getID() {
        std::lock_guard l(m);
        return id;
    }

    GPU& setID(int id) {
        std::lock_guard l(m);
        this->id = id;
        resetDevice();
        return *this;
    }

    GPU& resetDevice() {
        std::lock_guard l(m);
        HIPCHECK( hipSetDevice(id) );
        return *this;
    }

    int getCount() const {
        int gpucount;
        HIPCHECK( hipGetDeviceCount(&gpucount) );
        return gpucount;
    }

    void setmem(size_t mem) {
        if (mem > getmem()) Logger::warning(
            "Setting GPU memory to more than available ({:.1f} GB > {:.1f} GB)",
            mem / 1e9, getmem() / 1e9
        );
        this->mem = mem;
    }

    size_t getmem() const {
        if (mem > 0) return mem;

        hipDeviceProp_t prop {};
        HIPCHECK( hipGetDeviceProperties(&prop, id) );
        return prop.totalGlobalMem;
    }

private:
    GPU() = default;
    std::recursive_mutex m;
    int id {};
    size_t mem {};
};

template <typename T>
auto getKernelConfig(T fn, size_t N, size_t sharedMem=0) {
    static int nblocksmax, nthreads;

    [[maybe_unused]] static auto _ = [&]() {
        HIPCHECK( hipOccupancyMaxPotentialBlockSize(
            &nblocksmax, &nthreads, fn, sharedMem, 0
        ) );
        return true;
    }();

    return std::make_tuple(
        std::min<int>(nblocksmax, (N + nthreads - 1) / nthreads), nthreads
    );
};