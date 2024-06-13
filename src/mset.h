#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables.h>

#include "channel.h"
#include "coordinates.h"
#include "constants.h"
#include "logger.h"
#include "managedalloc.h"
#include "outputtypes.h"
#include "uvdatum.h"

class MeasurementSet {
public:
    enum class DataColumn {automatic, data, corrected, model};

    MeasurementSet() = default;

    MeasurementSet(
        const std::vector<std::string>& fnames,
        const DataColumn datacolumnname,
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

        // Load chosen datacolumn
        auto hascol = [] (const auto& ms, const auto& colname) -> bool {
            for (const auto& othercolname : ms.tableDesc().columnNames()) {
                if (casacore::String(colname) == othercolname) return true;
            }
            return false;
        };

        switch (datacolumnname) {
        case DataColumn::automatic:
            if (hascol(ms, "CORRECTED_DATA")) {
                Logger::info("Found CORRECTED_DATA column");
                datacolumn = casacore::MSMainColumns(ms).correctedData();
            } else if (hascol(ms, "DATA")) {
                Logger::info("No CORRECTED_DATA column found; using DATA instead");
                datacolumn = casacore::MSMainColumns(ms).data();
            } else {
                throw std::runtime_error(
                    "datacolumn=auto but neither DATA or CORRECTED_DATA exist"
                );
            }
            break;
        case DataColumn::data:
            if (hascol(ms, "DATA")) {
                datacolumn = casacore::MSMainColumns(ms).data();
            } else {
                throw std::runtime_error("DATA column not found");
            }
            break;
        case DataColumn::corrected:
            if (hascol(ms, "CORRECTED_DATA")) {
                datacolumn = casacore::MSMainColumns(ms).correctedData();
            } else {
                throw std::runtime_error("CORRECTED_DATA column not found");
            }
            break;
        case DataColumn::model:
            if (hascol(ms, "MODEL_DATA")) {
                datacolumn = casacore::MSMainColumns(ms).modelData();
            } else {
                throw std::runtime_error("MODEL_DATA column not found");
            }
            break;
        default:
            throw std::runtime_error("Invalid MeasurementSet::DataColumn value");
        }

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

        Logger::info(
            "Measurement set(s) opened: "
            "Channels {} - {} ({:.1f} - {:.1f} MHz) Times {:.0f} - {:.0f} selected",
            this->chanlow, this->chanhigh, freqs.front() / 1e6, freqs.back() / 1e6,
            this->timelow, this->timehigh
        );
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

    auto channelrange() const {
        return std::make_tuple(chanlow, chanhigh);
    }

    double midfreq() const {
        // TODO: Maybe (?) calculate midfreq using data weights?
        return std::accumulate(
            freqs.begin(), freqs.end(), 0.
        ) / freqs.size();
    }

    using FreqRange = std::tuple<double, double>;
    FreqRange freqrange() const {
        return std::make_tuple(freqs.front(), freqs.back());
    }

    double midchan() const {
        return (chanhigh + chanlow) / 2.;
    }

    double midtime() const {
        // Return mjd value (converted from seconds -> days)
        return (timehigh + timelow) / (2. * 86400.);
    }

    size_t size() const {
        auto [chanlo, chanhi] = this->channelrange();
        return ms.nrow() * (chanhi - chanlo + 1);
    }

    template <typename P, typename Alloc=ManagedAllocator<UVDatum<P>>>
    std::vector<UVDatum<P>, Alloc> data() const {
        // Pre-allocate the uvdata vector to avoid resizing operations
        // which MMapAllocator doesn't handle well
        std::vector<UVDatum<P>, Alloc> uvdata;
        uvdata.reserve(this->size());

        std::vector<double> lambdas = freqs;
        for (auto& x : lambdas) { x = Constants::c / x; };  // Convert to lambdas (m)

        casacore::MSMainColumns mscols(ms);

        // Pre-allocated array objects, since their memory allocation
        // can be reused between batches
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

        // Process nbatch rows at a time
        const long batchsize = 100;
        for (long nstart {}; nstart < static_cast<long>(ms.nrow()); nstart += batchsize) {
            // Note: nend is inclusive, so subtract 1
            long nend = std::min<long>(nstart + batchsize, ms.nrow()) - 1;

            // Set up slicers
            casacore::Slicer rowSlice {
                casacore::IPosition {nstart},
                casacore::IPosition {nend},
                casacore::Slicer::endIsLast
            };

            casacore::Slicer arraySlice {
                casacore::IPosition {0, chanlow},
                casacore::IPosition {3, chanhigh},
                casacore::Slicer::endIsLast
            };

            // Fetch row data in arrays
            mscols.uvw().getColumnRange(rowSlice, uvwArray, true);
            mscols.timeCentroid().getColumnRange(rowSlice, timeArray, true);
            mscols.antenna1().getColumnRange(rowSlice, ant1Array, true);
            mscols.antenna2().getColumnRange(rowSlice, ant2Array, true);
            mscols.weight().getColumnRange(rowSlice, weightArray, true);
            mscols.flagRow().getColumnRange(rowSlice, flagrowArray, true);
            mscols.flag().getColumnRange(rowSlice, arraySlice, flagArray, true);
            mscols.weightSpectrum().getColumnRange(rowSlice, arraySlice, weightspectrumArray, true);
            datacolumn.getColumnRange(rowSlice, arraySlice, dataArray, true);
            casacore::ArrayColumn<double>(fieldtbl, "PHASE_DIR")
                .getColumnRange(rowSlice, phaseCenterArray, true);

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
                        continue;
                    }

                    UVDatum<double> uvdatum {rowmeta, ncol, u, v, w, weights, data};
                    uvdata.push_back(static_cast<UVDatum<P>>(uvdatum));
                }  // chan iteration
            }  // nrow iteration
        }  // batch iteration

        // Sort (coarsely) by w value, thus keeping uvdata belonging to one w-layer in a
        // contiguous region of memory
        Logger::debug("Sorting uvdata by w value...");
        std::sort(uvdata.begin(), uvdata.end(), [] (auto& lhs, auto& rhs) {
            return static_cast<long>(lhs.w) < static_cast<long>(rhs.w);
        });
        Logger::debug("Sorting complete");

        return uvdata;
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
    casacore::ArrayColumn<std::complex<float>> datacolumn;
    int chanlow;
    int chanhigh;
    double timelow;  // stored as native mset value; i.e. mjd in _seconds_
    double timehigh; // stored as native mset value; i.e. mjd in _seconds_
};