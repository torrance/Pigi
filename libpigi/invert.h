#pragma once

#include "memory.h"
#include "workunit.h"

template <template<typename> typename T, typename S>
HostArray<T<S>, 2> invert(
    const HostSpan<WorkUnit<S>, 1> workunits,
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper,
    bool makePSF=false
);