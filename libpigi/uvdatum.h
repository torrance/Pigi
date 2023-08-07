#pragma once

#include "outputtypes.h"

template <typename T>
struct alignas(8) UVDatum {
    size_t row;
    size_t chan;
    T u;
    T v;
    T w;
    LinearData<T> weights;
    ComplexLinearData<T> data;

    explicit operator UVDatum<float>() const {
        return UVDatum<float>(
            row, chan, u, v, w,
            (LinearData<float>) weights,
            (ComplexLinearData<float>) data
        );
    }
};