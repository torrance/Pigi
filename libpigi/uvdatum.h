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

    template <typename S>
    explicit operator UVDatum<S>() const {
        return UVDatum<S> {
            row, chan, (S) u, (S) v, (S) w,
            (LinearData<S>) weights,
            (ComplexLinearData<S>) data
        };
    }
};