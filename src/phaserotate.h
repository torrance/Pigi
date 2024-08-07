#pragma once

#include <cmath>

#include "coordinates.h"
#include "datatable.h"
#include "timer.h"
#include "util.h"

void phaserotate(DataTable& tbl, const RaDec to) {
    auto timer = Timer::get("phaserotate");

    const RaDec from = tbl.phasecenter();
    if (to == from) return;

    const double cos_deltara = std::cos(to.ra - from.ra);
    const double sin_deltara = std::sin(to.ra - from.ra);
    const double sin_decfrom = std::sin(from.dec);
    const double cos_decfrom = std::cos(from.dec);
    const double sin_decto = std::sin(to.dec);
    const double cos_decto = std::cos(to.dec);

    for (size_t irow {}; auto& m : tbl.metadata()) {
        double u = m.u;
        double v = m.v;
        double w = m.w;

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

        m.u = uprime;
        m.v = vprime;
        m.w = wprime;

        // Add in geometric delay to data
        for (size_t ichan {}; const double lambda : tbl.lambdas()) {
            tbl.data(irow, ichan++) *= cispi(-2 * (wprime - w) / lambda);
        }
    }

    tbl.phasecenter(to);
}