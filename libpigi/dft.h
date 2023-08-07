#pragma once

#include "gridspec.h"
#include "memory.h"
#include "uvdatum.h"

template <typename T, typename S>
void idft(
    HostSpan<T, 2> img,
    HostSpan<UVDatum<S>, 1> uvdata,
    GridSpec gridspec,
    S normfactor
);