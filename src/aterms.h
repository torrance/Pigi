#pragma once

#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "beam.h"
#include "logger.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"

class Aterms {
public:
    using aterm_t = std::shared_ptr<HostArray<ComplexLinearData<double>, 2>>;

    // This constructor is only used in tests as a conversion from beam matrix to Aterms
    // object
    Aterms(const HostArray<ComplexLinearData<double>, 2>& aterm) {
        aterms.push_back(std::make_tuple(
            Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            },
            std::make_shared<HostArray<ComplexLinearData<double>, 2>>(aterm)
        ));
    }

    explicit
    Aterms(const std::vector<std::tuple<Interval, aterm_t>>& aterms) : aterms(aterms) {}

    const aterm_t get(double mjd, int) const {
        // Ignore antenna for now
        for (auto& [interval, aterm] : aterms) {
            if (interval.contains(mjd)) return aterm;
        }
        throw std::runtime_error(fmt::format("No Aterm found for time = {}", mjd));
    }

private:
    std::vector<std::tuple<Interval, aterm_t>> aterms {};
};

Aterms mkAterms(
    const casacore::MeasurementSet& mset,
    const GridSpec& gridspec,
    double maxDuration,
    const RaDec& gridorigin,
    const double freq
) {
    if (maxDuration <= 0) maxDuration = std::numeric_limits<double>::infinity();

    std::vector<std::tuple<Interval, typename Aterms::aterm_t>> aterms;

    std::string telescope = casacore::MSObservationColumns(
        mset.observation()
    ).telescopeName().get(0);

    // TODO: Remove these explicit branches on telescope type
    if (telescope == "MWA") {
        // Get MWA delays
        using mwadelay_t = std::tuple<double, double, std::array<uint32_t, 16>>;
        std::vector<mwadelay_t> delays;
        {
            auto mwaTilePointingTbl = mset.keywordSet().asTable("MWA_TILE_POINTING");
            auto intervalsCol = casacore::ArrayColumn<double>(mwaTilePointingTbl, "INTERVAL");
            auto delaysCol = casacore::ArrayColumn<int>(mwaTilePointingTbl, "DELAYS");

            for (size_t i {}; i < mwaTilePointingTbl.nrow(); ++i) {
                auto intervalRow = intervalsCol.get(i).tovector();
                auto delaysRow = delaysCol.get(i);

                // Copy casacore array to fixed array
                std::array<uint32_t, 16> delays_uint32;
                std::copy(delaysRow.begin(), delaysRow.end(), delays_uint32.begin());

                delays.push_back(std::make_tuple(
                    intervalRow[0] / 86400., intervalRow[1] / 86400., delays_uint32
                ));
            }
        }

        for (auto& [start, end, delays] : delays) {
            for (double t0 {start}; t0 < end; t0 += maxDuration) {
                double t1 = std::min(t0 + maxDuration, end);

                Interval interval(t0, t1);

                auto jones = Beam::MWA<double>(interval.mid(), delays).gridResponse(
                    gridspec, gridorigin, freq
                );
                aterms.push_back(std::make_tuple(
                    interval, std::make_shared<HostArray<ComplexLinearData<double>, 2>>(jones)
                ));
            }
        }
    } else {
        Logger::warning("Unkown telescope: defaulting to uniform beam");
        auto jones = Beam::Uniform<double>().gridResponse(gridspec, gridorigin, freq);
        aterms.push_back(std::make_tuple(
            Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            },
            std::make_shared<HostArray<ComplexLinearData<double>, 2>>(jones)
        ));
    }

    return Aterms{aterms};
}

template <template<typename> typename T, typename P>
HostArray<T<P>, 2> mkAvgAtermPower(const auto& workunits, const GridConfig& gridconf) {
    // Add together low-resolution beams at full double precision
    HostArray<T<double>, 2> beamPower64 {gridconf.subgrid().shape()};
    T<double> totalWeight {};

    for (auto& workunit : workunits) {
        T<double> weight = workunit.template totalWeight<P>();
        totalWeight += weight;

        for (size_t i {}, I = gridconf.subgrid().size(); i < I; ++i) {
            beamPower64[i] += static_cast<T<double>>(StokesI<P>::beamPower(
                (*workunit.Aleft)[i], (*workunit.Aright)[i]
            )) *= weight;
        }
    }

    // Normalise sum by its total weight
    for (auto& val : beamPower64) val /= totalWeight;

    // Now rescale and resize
    auto beamPower = static_cast<HostArray<T<P>, 2>>(beamPower64);
    beamPower = rescale(beamPower, gridconf.subgrid(), gridconf.padded());
    beamPower = resize(beamPower, gridconf.padded(), gridconf.grid());
    return beamPower;
}