#pragma once

#include "coordinates.h"
#include "outputtypes.h"
#include "sharedhostptr.h"

struct UVMeta {
    long row;
    double time;
    long ant1;
    long ant2;
    RaDec phasecenter;
};

template <typename T>
struct alignas(16) UVDatum {
    SharedHostPtr<UVMeta> meta;
    int chan;
    T u;
    T v;
    T w;
    LinearData<T> weights;
    ComplexLinearData<T> data;

    template <typename S>
    explicit operator UVDatum<S>() const {
        return UVDatum<S> {
            meta, chan,
            static_cast<S>(u), static_cast<S>(v), static_cast<S>(w),
            static_cast<LinearData<S>>(weights),
            static_cast<ComplexLinearData<S>>(data)
        };
    }
};