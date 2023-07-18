#pragma once

#include <complex>

#include "gridspec.cpp"
#include "outputtypes.cpp"

template <typename T>
struct UVWOrigin {
    T u0, v0, w0;

    UVWOrigin(T* ptr) : u0(ptr[0]), v0(ptr[1]), w0(ptr[2]) {}
    UVWOrigin(T u0, T v0, T w0) : u0(u0), v0(v0), w0(w0) {}
};

template <typename T>
struct WorkUnit {
    long long u0px;
    long long v0px;
    T u0;
    T v0;
    T w0;
    GridSpec subgridspec;
    SpanMatrix< Matrix2x2<std::complex<T>> > Aleft;
    SpanMatrix< Matrix2x2<std::complex<T>> > Aright;
    SpanVector< UVDatum<T> > data;
};