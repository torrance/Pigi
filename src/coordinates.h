#pragma once

#include <cmath>

#include <casacore/measures/Measures.h>
#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MPosition.h>

#include "util.h"

struct RaDec {
    double ra {};
    double dec {};

    bool operator==(const RaDec& ) const = default;
};

struct AzEl {
    double az {};  // Azimuth, measured North through to East (radians)
    double el {};  // Elevation angle, from horizon (radians)
};

struct LongLat {
    double lon {};
    double lat {};
};

RaDec lmToRaDec(const double l, const double m, const RaDec& origin) {
    // Don't use ndash here, as we _want_ below horizon values to be invalid
    // and for NaN to propagate
    auto n = std::sqrt(1 - l * l - m * m);

    auto ra = origin.ra + std::atan2(l, n * std::cos(origin.dec) - m * std::sin(origin.dec));
    auto dec = std::asin(m * std::cos(origin.dec) + n * std::sin(origin.dec));

    return {ra, dec};
}

auto RaDecTolm(const RaDec& radec, const RaDec& origin) {
    double l = std::cos(radec.dec) * std::sin(radec.ra - origin.ra);
    double m = std::sin(radec.dec) * std::cos(origin.dec) -
               std::cos(radec.dec) * std::sin(origin.dec) * std::cos(radec.ra - origin.ra);
    return std::make_tuple(l, m);
}

// Casacore doesn't seem to be threadsafe when doing coordinate conversions
thread_local std::mutex casacorelock;

AzEl radecToAzel(const RaDec& radec, const double& mjd, const LongLat& origin) {
    std::lock_guard l(casacorelock);

    // Create observation frame from time and origin
    casacore::MPosition pos(
        {{6378, "km"}, {origin.lon, "rad"}, {origin.lat, "rad"}}, casacore::MPosition::ITRF
    );

    casacore::MEpoch time({mjd, "d"}, casacore::MEpoch::UTC);

    // Create conversion object with from and to reference frames
    casacore::MDirection::Convert convert(
        casacore::MDirection::Ref {casacore::MDirection::J2000, {pos, time}},
        casacore::MDirection::Ref {casacore::MDirection::AZELNE, {pos, time}}
    );

    // Perform the actual conversion
    casacore::MDirection azel = convert(
        casacore::MDirection {{radec.ra, "rad"}, {radec.dec, "rad"}, casacore::MDirection::J2000}
    );

    return {azel.getValue().getLong(), azel.getValue().getLat()};
}