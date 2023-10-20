#pragma once

#include <thrust/complex.h>

#include "memory.h"
#include "mset.h"
#include "workunit.h"

template <template <typename> typename T, typename P>
struct ChannelGroup {
    int channelIndex;
    double midfreq;
    LinearData<P> weights;
    std::vector<MeasurementSet> msets;
    std::vector<WorkUnit<P>> workunits;
    HostArray<thrust::complex<P>, 2> psf;
    HostArray<T<P>, 2> residual;
    HostArray<T<P>, 2> components;
};