#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>

#include <casacore/ms/MeasurementSets.h>
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
            uvwCol(mset.ms, "UVW"),
            timeCol(mset.ms, "TIME_CENTROID"),
            ant1Col(mset.ms, "ANTENNA1"),
            ant2Col(mset.ms, "ANTENNA2"),
            flagrowCol(mset.ms, "FLAG_ROW"),
            flagCol(mset.ms, "FLAG"),
            weightCol(mset.ms, "WEIGHT"),
            weightspectrumCol(mset.ms, "WEIGHT_SPECTRUM"),
            dataCol(mset.ms, "CORRECTED_DATA"),
            phaseCenterCol(mset.fieldtbl, "PHASE_DIR") {

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
        casacore::ScalarColumn<double> timeCol;
        casacore::ScalarColumn<int> ant1Col;
        casacore::ScalarColumn<int> ant2Col;
        casacore::ScalarColumn<bool> flagrowCol;
        casacore::ArrayColumn<bool> flagCol;
        casacore::ArrayColumn<float> weightCol;
        casacore::ArrayColumn<float> weightspectrumCol;
        casacore::ArrayColumn<std::complex<float>> dataCol;
        casacore::ArrayColumn<double> phaseCenterCol;

        casacore::Array<double> uvwArray;
        casacore::Vector<double> timeArray;
        casacore::Vector<int> ant1Array;
        casacore::Vector<int> ant2Array;
        casacore::Vector<bool> flagrowArray;
        casacore::Array<bool> flagArray;
        casacore::Array<float> weightArray;
        casacore::Array<float> weightspectrumArray;
        casacore::Array<std::complex<float>> dataArray;
        casacore::Array<double> phaseCenterArray;

        void rebuild_cache() {
            // Clear the cache
            cache.clear();

            // Hardcode the batchsize for now
            const long batchsize = 100;

            // Early return if we've finished the table
            if (nstart == -1 || nstart >= static_cast<long>(mset.ms.nrow())) {
                nstart = -1;
                return;
            }

            // Note: nend is inclusive, so subtract 1
            long nend = std::min(
                nstart + batchsize,
                static_cast<long>(mset.ms.nrow())
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
            timeCol.getColumnRange(rowSlice, timeArray, true);
            ant1Col.getColumnRange(rowSlice, ant1Array, true);
            ant2Col.getColumnRange(rowSlice, ant2Array, true);
            weightCol.getColumnRange(rowSlice, weightArray, true);
            flagrowCol.getColumnRange(rowSlice, flagrowArray, true);
            flagCol.getColumnRange(rowSlice, arraySlice, flagArray, true);
            weightspectrumCol.getColumnRange(rowSlice, arraySlice, weightspectrumArray, true);
            dataCol.getColumnRange(rowSlice, arraySlice, dataArray, true);
            phaseCenterCol.getColumnRange(rowSlice, phaseCenterArray, true);

            // Create iterators
            auto uvwIter = uvwArray.begin();
            auto timeIter = timeArray.begin();
            auto ant1Iter = ant1Array.begin();
            auto ant2Iter = ant2Array.begin();
            auto flagrowIter = flagrowArray.begin();
            auto weightIter = weightArray.begin();
            auto dataIter = dataArray.begin();
            auto weightspectrumIter = weightspectrumArray.begin();
            auto flagIter = flagArray.begin();
            auto phaseCenterIter = phaseCenterArray.begin();

            for (long nrow {nstart}; nrow <= nend; ++nrow) {
                double u_m = *uvwIter; ++uvwIter;
                double v_m = -(*uvwIter); ++uvwIter;  // Invert v for MWA observations only (?)
                double w_m = *uvwIter; ++uvwIter;

                double time = *timeIter / 86400.; ++timeIter;
                int ant1 = *ant1Iter; ++ant1Iter;
                int ant2 = *ant2Iter; ++ant2Iter;

                bool flagrow = *flagrowIter; ++flagrowIter;

                double ra0 = *phaseCenterIter; ++phaseCenterIter;
                double dec0 = *phaseCenterIter; ++phaseCenterIter;

                auto rowmeta = makesharedhost<UVMeta>(
                    nrow, time, ant1, ant2, RaDec{ra0, dec0}
                );

                for (int ncol {}; ncol < static_cast<int>(lambdas.size()); ++ncol) {
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
                        rowmeta, ncol, u, v, w, weights, data
                    });
                }  // chan iteration
            }  // nrow iteration

            cache_current = cache.begin(); // Reset iterator, as it may have changed
            nstart += batchsize;
        }
    };

    MeasurementSet() = default;

    MeasurementSet(
        const std::vector<std::string>& fnames,
        const int chanlow,
        const int chanhigh,
        const double timelow = std::numeric_limits<double>::min(),
        const double timehigh = std::numeric_limits<double>::max()
    ) : chanlow(chanlow), chanhigh(chanhigh), timelow(timelow), timehigh(timehigh) {
        // Concatenate tables, whilst preserving FIELD_ID correctness
        {
            casacore::Block<casacore::String> fnamesblock(fnames.size());
            for (size_t i {}; i < fnames.size(); ++i) {
                fnamesblock[i] = fnames[i];
            }

            // Concatenate the following tables; the default is to use
            // only the first table's subtables.
            casacore::Block<casacore::String> concats(3);
            concats[0] = "FIELD";
            concats[1] = "OBSERVATION";
            concats[2] = "MWA_TILE_POINTING";

            tbls = casacore::Table(fnamesblock, concats);
            tbls.rename(std::tmpnam(NULL), casacore::Table::New);
            tbls.flush();

            // Once concatenated, we can create a reference table that forwards columns
            // to the concatenated table. The exceptions are index columns to concatenated
            // subtables, which we need to update.
            casacore::Block<casacore::String> freshcols(2);
            freshcols[0] = "FIELD_ID";
            freshcols[1] = "OBSERVATION_ID";
            tbl = casacore::MeasurementSet(tbls).referenceCopy(std::tmpnam(NULL), freshcols);

            // Copy tbl -> ms, so that we can delete the original table despite later
            // filtering the tbl.
            ms = tbl;

            long mainOffset {}, fieldIdOffset {}, observationIdOffset {};
            for (auto& fname : fnames) {
                casacore::MeasurementSet ms0(fname);

                // Increment the field ID by the offset
                auto fieldIds = casacore::MSMainColumns(ms0).fieldId().getColumn();
                for (auto& fieldId : fieldIds) fieldId += fieldIdOffset;
                fieldIdOffset += ms0.field().nrow();

                // And write updated ID back into concatenated field_id column
                casacore::MSMainColumns(ms).fieldId().putColumnRange(
                    {casacore::IPosition(1, mainOffset), casacore::IPosition(1, ms0.nrow())},
                    fieldIds
                );

                // Increment the observation ID by the offset
                auto observationIds = casacore::MSMainColumns(ms0).observationId().getColumn();
                for (auto& observationId : observationIds) observationId += observationIdOffset;
                observationIdOffset += ms0.observation().nrow();

                // And write updated ID back into concatenated observation_id column
                casacore::MSMainColumns(ms).observationId().putColumnRange(
                    {casacore::IPosition(1, mainOffset), casacore::IPosition(1, ms0.nrow())},
                    observationIds
                );

                mainOffset += ms0.nrow();
            }
        }

        // Filter table by time low and high
        ms = ms(
            ms.col("TIME_CENTROID") >= timelow &&
            ms.col("TIME_CENTROID") <= timehigh
        );

        // Remove auto-correlations
        ms = ms(
            ms.col("ANTENNA1") != ms.col("ANTENNA2")
        );

        // Remove flagged row
        ms = ms(
            !ms.col("FLAG_ROW")
        );

        fieldtbl = casacore::tableCommand(
            "SELECT t2.PHASE_DIR FROM $1 t1 "
            "JOIN ::FIELD t2 ON t1.FIELD_ID = msid(t2.FIELD_ID)",
            ms
        ).table();

        // Set actual timelow and timehigh
        casacore::Vector<double> timeCol(
            (casacore::ScalarColumn<double> {ms, "TIME_CENTROID"}).getColumn()
        );
        auto minmax = std::minmax_element(timeCol.begin(), timeCol.end());
        this->timelow = *std::get<0>(minmax);
        this->timehigh = *std::get<1>(minmax);

        // Get channel / freq information
        // We assume the spectral window is identical for all
        auto subtbl = ms.keywordSet().asTable({"SPECTRAL_WINDOW"});
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

        fmt::println("Measurement set(s) opened");
        fmt::println(
            "   Channels {} - {} ({:.1f} - {:.1f} MHz) Times {:.0f} - {:.0f} selected from",
            this->chanlow, this->chanhigh, freqs.front() / 1e6, freqs.back() / 1e6,
            this->timelow, this->timehigh
        );
        for (auto& fname : fnames) {
            fmt::println("      - {}", fname);
        }
    }

    MeasurementSet(const MeasurementSet&) = delete;
    MeasurementSet(MeasurementSet&&) = delete;
    MeasurementSet& operator=(const MeasurementSet&) = delete;
    MeasurementSet& operator=(MeasurementSet&&) = delete;
    ~MeasurementSet() {
        // For some reason, these can't be marked for deletion at the time of creation
        // I'm not sure why, but otherwise a 'Table does not exist' exception is thrown.
        tbl.markForDelete();
        tbls.markForDelete();
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
        auto names = casacore::MSObservationColumns(
            ms.observation()
        ).telescopeName().getColumn();

        // TODO: Handle more than one type of telescope?
        for (auto& name : names) {
            if (name != names[0]) {
                throw std::runtime_error("Multiple telescope types detected");
            }
        }

        return names[0];
    }

    auto mwaDelays() const {
        auto mwaTilePointingTbl = ms.keywordSet().asTable("MWA_TILE_POINTING");
        auto intervals = casacore::ArrayColumn<double>(mwaTilePointingTbl, "INTERVAL");
        auto delays = casacore::ArrayColumn<int>(mwaTilePointingTbl, "DELAYS");

        using mwadelay_t = std::tuple<double, double, std::array<uint32_t, 16>>;

        std::vector<mwadelay_t> delays_vec;

        for (size_t i {}; i < mwaTilePointingTbl.nrow(); ++i) {
            auto interval_row = intervals.get(i).tovector();
            auto delays_row = delays.get(i);

            // Copy casacore array to fixed array
            std::array<uint32_t, 16> delays_uint32;
            std::copy(delays_row.begin(), delays_row.end(), delays_uint32.begin());

            delays_vec.push_back(std::make_tuple(
                interval_row[0] / 86400., interval_row[1] / 86400., delays_uint32
            ));
        }

        return delays_vec;
    }

    RaDec phaseCenter() const {
        auto sourceTbl = ms.keywordSet().asTable("FIELD");
        auto direction = casacore::ArrayColumn<double>(sourceTbl, "PHASE_DIR").get(0);
        return {
            direction(casacore::IPosition {0, 0}), direction(casacore::IPosition {1, 0})
        };
    }

private:
    casacore::Table tbls;
    casacore::Table tbl;
    casacore::MeasurementSet ms;
    casacore::Table fieldtbl;
    std::vector<double> freqs;
    int chanlow;
    int chanhigh;
    double timelow;  // stored as native mset value; i.e. mjd in _seconds_
    double timehigh; // stored as native mset value; i.e. mjd in _seconds_
};