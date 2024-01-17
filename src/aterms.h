#pragma once

#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "beam.h"
#include "memory.h"
#include "mset.h"
#include "outputtypes.h"
#include "util.h"

template <typename P>
class Aterms {
public:
    using aterm_t = std::shared_ptr<HostArray<ComplexLinearData<P>, 2>>;

    // This constructor is only used in tests as a conversion from beam matrix to Aterms
    // object
    Aterms(const HostArray<ComplexLinearData<P>, 2>& aterm) {
        aterms.push_back(std::make_tuple(
            Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            },
            std::make_shared<HostArray<ComplexLinearData<P>, 2>>(aterm)
        ));
    }

    explicit
    Aterms(const std::vector<std::tuple<Interval, aterm_t>>& aterms) : aterms(aterms) {}

    const aterm_t get(double mjd, int) const {
        // Ignore antenna for now
        for (auto& [interval, aterm] : aterms) {
            if (interval.contains(mjd)) return aterm;
        }
        throw std::runtime_error("No Aterm found for specified time");
    }

private:
    std::vector<std::tuple<Interval, aterm_t>> aterms {};
};

template <typename P>
Aterms<P> mkAterms(
    MeasurementSet& mset,
    const GridSpec& gridspec,
    const double maxDuration,
    RaDec& gridorigin
) {
    std::vector<std::tuple<Interval, typename Aterms<P>::aterm_t>> aterms;

    // TODO: Remove these explicit branches on telescope type
    if (mset.telescopeName() == "MWA") {
        for (auto& [start, end, delays] : mset.mwaDelays()) {
            int Nintervals = 1;
            if (maxDuration > 0) {
                Nintervals = std::ceil((end - start) / (maxDuration / 86400.));
            }

            // Break it up into N evenly-sized intervals
            double width = (end - start) / Nintervals;

            for (int n {}; n < Nintervals; ++n) {
                Interval interval(start + n * width, start + (n + 1) * width);

                auto jones = Beam::MWA<P>(interval.mid(), delays).gridResponse(
                    gridspec, gridorigin, mset.midfreq()
                );
                aterms.push_back(std::make_tuple(
                    interval, std::make_shared<HostArray<ComplexLinearData<P>, 2>>(jones)
                ));
            }
        }
    } else {
        fmt::println("Unkown telescope: defaulting to uniform beam");
        auto jones = Beam::Uniform<P>().gridResponse(gridspec, gridorigin, mset.midfreq());
        aterms.push_back(std::make_tuple(
            Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            },
            std::make_shared<HostArray<ComplexLinearData<P>, 2>>(jones)
        ));
    }

    return Aterms{aterms};
}

template <template<typename> typename T, typename P>
HostArray<T<P>, 2> mkAvgAtermPower(auto&& workunits, const GridConfig& gridconf) {
    // Add together low-resolution beams at full double precision
    HostArray<T<double>, 2> beamPower64 {gridconf.subgrid().shape()};
    T<double> totalWeight {};

    for (auto& workunit : workunits) {
        T<double> weight = workunit.template totalWeight<P>();
        totalWeight += weight;

        for (size_t i {}, I = gridconf.subgrid().size(); i < I; ++i) {
            beamPower64[i] += static_cast<T<double>>(StokesI<P>::beamPower(
                workunit.Aleft->operator[](i), workunit.Aright->operator[](i)
            )) *= weight;
        }
    }

    // Normalise sum by its total weight
    for (auto& val : beamPower64) val /= totalWeight;

    // No rescale and resize
    auto beamPower = static_cast<HostArray<T<P>, 2>>(beamPower64);
    beamPower = rescale(beamPower, gridconf.subgrid(), gridconf.padded());
    beamPower = resize(beamPower, gridconf.padded(), gridconf.grid());

    return beamPower;
}