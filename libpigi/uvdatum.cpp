#pragma once

#include <hip/hip_runtime.h>

#include "outputtypes.cpp"

template <typename T>
struct UVDatum {
    size_t row;
    size_t chan;
    T u;
    T v;
    T w;
    LinearData<T> weights;
    ComplexLinearData<T> data;
};