#pragma once

#include <cmath>

#include "coordinates.h"
#include "uvdatum.h"
#include "util.h"

template<typename P>
void phaserotate(UVDatum<P>& uvdatum, const RaDec to) {
    const RaDec from = uvdatum.meta->phasecenter;

    const double cos_deltara = std::cos(to.ra - from.ra);
    const double sin_deltara = std::sin(to.ra - from.ra);
    const double sin_decfrom = std::sin(from.dec);
    const double cos_decfrom = std::cos(from.dec);
    const double sin_decto = std::sin(to.dec);
    const double cos_decto = std::cos(to.dec);

    const double u {uvdatum.u}, v {uvdatum.v}, w {uvdatum.w};

    const double uprime = (
        + u * cos_deltara
        - v * sin_decfrom * sin_deltara
        - w * cos_decfrom * sin_deltara
    );
    const double vprime = (
        + u * sin_decto * sin_deltara
        + v * (sin_decfrom * sin_decto * cos_deltara + cos_decfrom * cos_decto)
        - w * (sin_decfrom * cos_decto - cos_decfrom * sin_decto * cos_deltara)
    );
    const double wprime = (
        + u * cos_decto * sin_deltara
        - v * (cos_decfrom * sin_decto - sin_decfrom * cos_decto * cos_deltara)
        + w * (sin_decfrom * sin_decto + cos_decfrom * cos_decto * cos_deltara)
    );

    uvdatum.u = uprime;
    uvdatum.v = vprime;
    uvdatum.w = wprime;
    uvdatum.data *= cispi(-2 * (wprime - w));
}