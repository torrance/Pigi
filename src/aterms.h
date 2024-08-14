#pragma once

#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/functional/hash.hpp>

#include "beam.h"
#include "logger.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"
#include "workunit.h"

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

    template <template<typename> typename T, typename P>
    HostArray<T<P>, 2> average(
        DataTable& tbl, std::vector<WorkUnit>& workunits, GridConfig gridconf
    ) {
        // All intermediate weights and beams are calculated at full double precision
        // to avoid FP errors with large weight sums.

        auto subgridspec = gridconf.subgrid();
        auto gridspec = gridconf.grid();
        auto paddedspec = gridconf.padded();

        // Accumulate data weights per Aterm
        using Key = std::pair<aterm_t, aterm_t>;
        struct KeyHasher {
            std::size_t operator()(const Key& key) const {
                size_t seed {};
                boost::hash_combine(seed, std::get<0>(key));
                boost::hash_combine(seed, std::get<1>(key));
                return seed;
            }
        };
        std::unordered_map<Key, T<double>, KeyHasher> atermweights;

        for (auto& w : workunits) {
            auto [ant1, ant2] = w.baseline;
            aterm_t aleft = this->get(w.time, ant1);
            aterm_t aright = this->get(w.time, ant2);

            // Add wieght contribution for this pair of beams
            T<double>& weight = atermweights[{aleft, aright}];
            for (size_t irow {w.rowstart}; irow < w.rowend; ++irow) {
                for (size_t ichan {w.chanstart}; ichan < w.chanend; ++ichan) {
                    weight += static_cast<T<double>>(tbl.weights(irow, ichan));
                }
            }
        }

        // Calculate weight to normalize average beam
        T<double> totalWeight {};
        for (auto [atermpair, weight] : atermweights) totalWeight += weight;

        // Now for each baseline pair, compute the power and append to total beam power
        HostArray<T<double>, 2> beamPower64 {subgridspec.shape()};
        for (auto [atermpair, weight] : atermweights) {
            auto [aleft, aright] = atermpair;

            for (size_t i {}, I = subgridspec.size(); i < I; ++i) {
                beamPower64[i] += T<double>::beamPower(
                    (*aleft)[i], (*aright)[i]
                ) *= (weight /= totalWeight);
            }
        }

        // Now rescale and resize
        auto beamPower = static_cast<HostArray<T<P>, 2>>(beamPower64);
        beamPower = rescale(beamPower, subgridspec, paddedspec);
        beamPower = resize(beamPower, paddedspec, gridspec);
        return beamPower;
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