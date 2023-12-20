#pragma once

#include <algorithm>
#include <map>
#include <limits>
#include <string>
#include <utility>

#include <casacore/tables/Tables.h>

#include "channel.h"
#include "coordinates.h"
#include "constants.h"
#include "outputtypes.h"
#include "uvdatum.h"

class MeasurementSet {
public:
    class Iterator {
    public:
        Iterator(MeasurementSet& mset, long nstart = 0) :
            mset(mset), lambdas(mset.freqs), nstart(nstart),
            uvwCol(mset.tbl, "UVW"),
            flagrowCol(mset.tbl, "FLAG_ROW"),
            flagCol(mset.tbl, "FLAG"),
            weightCol(mset.tbl, "WEIGHT"),
            weightspectrumCol(mset.tbl, "WEIGHT_SPECTRUM"),
            dataCol(mset.tbl, "CORRECTED_DATA") {

            // Ensure our expectations about the size of cells are valid.
            // Cells selected using the slice are guaranteed to be correct and we don't need to
            // check again here.
            if (uvwCol.shape(0) != casacore::IPosition{3}) abort();
            if (weightCol.shape(0) != casacore::IPosition {4}) abort();

            // Convert freqs -> lambdas
            for (auto& x : lambdas) { x = Constants::c / x; };  // Convert to lambdas (m)

            rebuild_cache();
        }

        auto& operator*() const { return *cache_current; }

        auto& operator++() {
            if (++cache_current == cache.end()) rebuild_cache();
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return nstart == other.nstart;
        }

        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        MeasurementSet& mset;
        std::vector<double> lambdas;
        std::vector<UVDatum<double>> cache;
        std::vector<UVDatum<double>>::iterator cache_current;
        long nstart {};

        casacore::ArrayColumn<double> uvwCol;
        casacore::ScalarColumn<bool> flagrowCol;
        casacore::ArrayColumn<bool> flagCol;
        casacore::ArrayColumn<float> weightCol;
        casacore::ArrayColumn<float> weightspectrumCol;
        casacore::ArrayColumn<std::complex<float>> dataCol;

        casacore::Array<double> uvwArray;
        casacore::Vector<bool> flagrowArray;
        casacore::Array<bool> flagArray;
        casacore::Array<float> weightArray;
        casacore::Array<float> weightspectrumArray;
        casacore::Array<std::complex<float>> dataArray;

        void rebuild_cache() {
            // Clear the cache
            cache.clear();

            // Hardcode the batchsize for now
            const long batchsize = 100;

            // Early return if we've finished the table
            if (nstart == -1 || nstart >= static_cast<long>(mset.tbl.nrow())) {
                nstart = -1;
                return;
            }

            // Note: nend is inclusive, so subtract 1
            long nend = std::min(
                nstart + batchsize,
                static_cast<long>(mset.tbl.nrow())
            ) - 1;

            // Set up slicers
            casacore::Slicer rowSlice {
                casacore::IPosition {nstart},
                casacore::IPosition {nend},
                casacore::Slicer::endIsLast
            };

            casacore::Slicer arraySlice {
                casacore::IPosition {0, mset.chanlow},
                casacore::IPosition {3, mset.chanhigh},
                casacore::Slicer::endIsLast
            };

            // Fetch row data in arrays
            uvwCol.getColumnRange(rowSlice, uvwArray, true);
            weightCol.getColumnRange(rowSlice, weightArray, true);
            flagrowCol.getColumnRange(rowSlice, flagrowArray, true);
            flagCol.getColumnRange(rowSlice, arraySlice, flagArray, true);
            weightspectrumCol.getColumnRange(rowSlice, arraySlice, weightspectrumArray, true);
            dataCol.getColumnRange(rowSlice, arraySlice, dataArray, true);

            // Create iterators
            auto uvwIter = uvwArray.begin();
            auto flagrowIter = flagrowArray.begin();
            auto weightIter = weightArray.begin();
            auto dataIter = dataArray.begin();
            auto weightspectrumIter = weightspectrumArray.begin();
            auto flagIter = flagArray.begin();

            for (long nrow {nstart}; nrow <= nend; ++nrow) {
                double u_m = *uvwIter; ++uvwIter;
                double v_m = -(*uvwIter); ++uvwIter;  // Invert v for MWA observations only (?)
                double w_m = *uvwIter; ++uvwIter;

                bool flagrow = *flagrowIter; ++flagrowIter;

                for (size_t ncol {}; ncol < lambdas.size(); ++ncol) {
                    double u = u_m / lambdas[ncol];
                    double v = v_m / lambdas[ncol];
                    double w = w_m / lambdas[ncol];

                    LinearData<double> weights;
                    weights.xx = *weightspectrumIter; ++weightspectrumIter;
                    weights.xy = *weightspectrumIter; ++weightspectrumIter;
                    weights.yx = *weightspectrumIter; ++weightspectrumIter;
                    weights.yy = *weightspectrumIter; ++weightspectrumIter;

                    // Negate the flags so that flagged data = 0 when used as a weight
                    LinearData<bool> flags;
                    flags.xx = !*flagIter; ++flagIter;
                    flags.xy = !*flagIter; ++flagIter;
                    flags.yx = !*flagIter; ++flagIter;
                    flags.yy = !*flagIter; ++flagIter;

                    // Flags have the effect to set weights to 0
                    (weights *= !flagrow) *= flags;

                    ComplexLinearData<double> data;
                    data.xx = *dataIter; ++dataIter;
                    data.xy = *dataIter; ++dataIter;
                    data.yx = *dataIter; ++dataIter;
                    data.yy = *dataIter; ++dataIter;

                    if (!weights.isfinite() || !data.isfinite()) {
                        data = {};
                        weights = {};
                    }

                    // We can always force w >= 0, since V(u, v, w) = V*(-u, -v, -w)
                    // and this helps reduce the number of distinct w-layers.
                    if (w < 0) {
                        u = -u; v = -v; w = -w;
                        data = data.adjoint();
                        weights = weights.adjoint();
                    }

                    cache.push_back(UVDatum<double> {
                        static_cast<size_t>(nrow), ncol, u, v, w, weights, data
                    });
                }  // chan iteration
            }  // nrow iteration

            cache_current = cache.begin(); // Reset iterator, as it may have changed
            nstart += batchsize;
        }
    };

    MeasurementSet() = default;

    MeasurementSet(
        const std::string fname,
        const int chanlow,
        const int chanhigh,
        const double timelow = std::numeric_limits<double>::min(),
        const double timehigh = std::numeric_limits<double>::max()
    ) : tbl(fname),
        chanlow(chanlow), chanhigh(chanhigh), timelow(timelow), timehigh(timehigh) {

        // Filter table by time low and high
        tbl = tbl(
            tbl.col("TIME_CENTROID") >= timelow &&
            tbl.col("TIME_CENTROID") <= timehigh
        );

        // Set actual timelow and timehigh
        casacore::Vector<double> timeCol(
            (casacore::ScalarColumn<double> {tbl, "TIME_CENTROID"}).getColumn()
        );
        auto minmax = std::minmax_element(timeCol.begin(), timeCol.end());
        this->timelow = *std::get<0>(minmax);
        this->timehigh = *std::get<1>(minmax);

        // Get channel / freq information
        auto subtbl = tbl.keywordSet().asTable({"SPECTRAL_WINDOW"});
        casacore::ArrayColumn<double> freqsCol(subtbl, "CHAN_FREQ");
        freqs = freqsCol.get(0).tovector();

        // Set actual chanhigh
        this->chanhigh = std::min<int>(this->chanhigh, freqs.size() - 1);
        if (this->chanhigh == -1) {
            this->chanhigh = freqs.size() - 1;
        }

        // Trim the freqs to the actual channel range
        freqs = freqsCol.getSlice(0, casacore::Slicer{
            casacore::IPosition {this->chanlow}, casacore::IPosition {this->chanhigh},
            casacore::Slicer::endIsLast
        }).tovector();

        fmt::println("Measurement set {} opened", fname);
        fmt::println(
            "    Channels {} - {} ({:.1f} - {:.1f} MHz) Times {:.0f} - {:.0f} selected",
            this->chanlow, this->chanhigh, freqs.front() / 1e6, freqs.back() / 1e6,
            this->timelow, this->timehigh
        );
    }

    auto begin() { return Iterator(*this); }
    auto end() { return Iterator(*this, -1); }

    auto channelrange() {
        return std::make_tuple(chanlow, chanhigh);
    }

    double midfreq() const {
        // TODO: Maybe (?) calculate midfreq using data weights?
        return std::accumulate(
            freqs.begin(), freqs.end(), 0.
        ) / freqs.size();
    }

    double midchan() const {
        return (chanhigh + chanlow) / 2.;
    }

    double midtime() const {
        // Return mjd value (converted from seconds -> days)
        return (timehigh + timelow) / (2. * 86400.);
    }

    std::string telescopeName() const {
        auto observationTbl = tbl.keywordSet().asTable("OBSERVATION");
        return casacore::ScalarColumn<casacore::String>(
            observationTbl, "TELESCOPE_NAME"
        ).get(0);
    }

    std::array<uint32_t, 16> mwaDelays() const {
        auto mwaTilePointingTbl = tbl.keywordSet().asTable("MWA_TILE_POINTING");
        auto delays = casacore::ArrayColumn<int>(
            mwaTilePointingTbl, "DELAYS"
        ).get(0);

        // Copy casacore array to
        std::array<uint32_t, 16> delays_uint32;
        std::copy(
            delays.begin(), delays.end(), delays_uint32.begin()
        );

        return delays_uint32;
    }

    RaDec phaseCenter() const {
        auto sourceTbl = tbl.keywordSet().asTable("FIELD");
        auto direction = casacore::ArrayColumn<double>(sourceTbl, "PHASE_DIR").get(0);
        return {
            direction(casacore::IPosition {0, 0}), direction(casacore::IPosition {1, 0})
        };
    }

    static std::map<int, std::vector<MeasurementSet>> partition(
        std::vector<std::string> fnames,
        int chanlow, int chanhigh,
        int channelsOut = 1,
        double maxDuration = std::numeric_limits<double>::max()
    ) {
        std::map<int, std::vector<MeasurementSet>> msets;

        // ASSUME: All spectral indices match the first table
        // TODO: Pass in chanlo and chanhi
        auto spectralWindowTbl = casacore::Table(fnames[0])
            .keywordSet().asTable("SPECTRAL_WINDOW");

        auto freqs = casacore::ArrayColumn<double>(spectralWindowTbl, "CHAN_FREQ")
            .get(0).tovector();

        // Ensure chanlow >= 0
        chanlow = std::max(0, chanlow);

        // If chan is set > than the number of channels, assume we want up to max channels
        chanhigh = std::min<int>(chanhigh, freqs.size() - 1);

        auto chanlength = chanhigh - chanlow + 1;  // +1 since chanhigh is inclusive

        // channelBounds contains pairs of low and high channels (inclusive)
        std::vector<std::pair<int, int>> channelBounds {0};
        for (
            int chanBoundLow {};
            chanBoundLow < chanhigh;
            chanBoundLow += std::ceil(chanlength / channelsOut)
        ) {
            int chanBoundHigh = std::min<int>(
                chanBoundLow + std::ceil(chanlength / channelsOut) - 1, chanhigh
            );
            channelBounds.push_back({chanBoundLow, chanBoundHigh});
        }

        for (auto fname : fnames) {
            // ASSUME: One observation per mset
            // TODO: Fix this
            auto observationTbl = casacore::Table(fname).keywordSet().asTable("OBSERVATION");
            auto timeRange = casacore::ArrayColumn<double>(observationTbl, "TIME_RANGE")
                .get(0).tovector();

            double duration = (timeRange[1] - timeRange[0]);
            int timesOut = std::ceil(duration / maxDuration);

            // timelow <= x < timehigh
            for (int tn {}; tn < timesOut; ++tn) {
                double timelow = tn * duration / timesOut + timeRange[0];
                double timehigh = (tn + 1) * duration / timesOut + timeRange[0];

                // With the exception of the last interval, we don't want the upper bound
                // to be inclusive.
                timehigh = std::nexttoward(timehigh, 0.);

                // Avoid floating point errors, since we are testing using equality
                if (tn == timesOut - 1) timehigh = timeRange[1];

                for (int channelIndex {}; auto [chanBoundLow, chanBoundHigh] : channelBounds) {
                    msets[++channelIndex].emplace_back(
                        fname, chanBoundLow, chanBoundHigh, timelow, timehigh
                    );

                    fmt::println(
                        "Opened mset {} channels {} - {}, time {} - {}",
                        fname, chanBoundLow, chanBoundHigh, timelow, timehigh
                    );
                }

            }
        }

        return msets;
    }

private:
    casacore::Table tbl;
    std::vector<double> freqs;
    int chanlow;
    int chanhigh;
    double timelow;  // stored as native mset value; i.e. mjd in _seconds_
    double timehigh; // stored as native mset value; i.e. mjd in _seconds_
};