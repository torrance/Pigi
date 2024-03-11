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
    SharedHostPtr<UVMeta> meta {};
    int chan {};
    T u {};
    T v {};
    T w {};
    LinearData<T> weights {};
    ComplexLinearData<T> data {};

    UVDatum() = default;

    UVDatum(
        SharedHostPtr<UVMeta> meta, int chan,
        T u, T v, T w,
        LinearData<T> weights, ComplexLinearData<T> data
    ) : meta(meta), chan(chan), u(u), v(v), w(w),  weights(weights), data(data){
        forcePositiveW();
    }

    template <typename S>
    explicit operator UVDatum<S>() const {
        return UVDatum<S> {
            meta, chan,
            static_cast<S>(u), static_cast<S>(v), static_cast<S>(w),
            static_cast<LinearData<S>>(weights),
            static_cast<ComplexLinearData<S>>(data)
        };
    }

    UVDatum& forcePositiveW() {
        // Since V(u, v, w) = V(-u, -v, -w)*, we can force w to be positive
        // This reduces the number of w-layers and thus processing time
        if (w < 0) {
            this->u = -u;
            this->v = -v;
            this->w = -w;
            this->weights = weights.adjoint();
            this->data = data.adjoint();
        }
        return *this;
    }
};