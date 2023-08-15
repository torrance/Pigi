#include "findabsmax.h"

#include <iostream>

#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

template <typename T>
std::tuple<size_t, std::complex<T>> findabsmax(std::complex<T>* first, std::complex<T>* last) {
    std::complex<T>* maxptr = thrust::max_element(thrust::device, first, last, [](auto lhs, auto rhs) {
        return std::abs(lhs.real()) < std::abs(rhs.real());
    });

    size_t idx = maxptr - first;

    std::complex<T> maxval;
    [[maybe_unused]] auto _ = hipMemcpyDtoHAsync(&maxval, maxptr, sizeof(std::complex<T>), hipStreamPerThread);
    _ = hipStreamSynchronize(hipStreamPerThread);

    return std::make_tuple(idx, maxval);
}

template std::tuple<size_t, std::complex<float>> findabsmax(std::complex<float>*, std::complex<float>*);
template std::tuple<size_t, std::complex<double>> findabsmax(std::complex<double>*, std::complex<double>*);