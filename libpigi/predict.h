#pragma once

#include "degridop.h"
#include "memory.h"
#include "workunit.h"

template <typename T, typename S>
void predict(
    HostSpan<WorkUnit<S>, 1> workunits,
    const HostSpan<T, 2> img,
    const GridSpec gridspec,
    const HostSpan<S, 2> taper,
    const HostSpan<S, 2> subtaper,
    const DegridOp degridop=DegridOp::Replace
);