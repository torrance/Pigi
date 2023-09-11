#pragma once

#include <string>

#include <casacore/tables/Tables.h>

#include "channel.h"
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
                double v_m = *uvwIter; ++uvwIter;
                double w_m = *uvwIter; ++uvwIter;

                bool flagrow = *flagrowIter; ++flagrowIter;

                LinearData<double> weightRow;
                weightRow.xx = *weightIter; ++weightIter;
                weightRow.xy = *weightIter; ++weightIter;
                weightRow.yx = *weightIter; ++weightIter;
                weightRow.yy = *weightIter; ++weightIter;
                weightRow *= !flagrow;  // Flagged row has the effect to set all to zero

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

                    (weights *= weightRow) *= flags;

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

    struct Config {
        int chanlow {-1};
        int chanhigh {-1};
    };

    MeasurementSet(const std::string fname, const Config config) :
        tbl(fname), chanlow(config.chanlow), chanhigh(config.chanhigh) {

        // Get channel / freq information
        auto subtbl = tbl.keywordSet().asTable({"SPECTRAL_WINDOW"});
        if (chanlow == -1 || chanhigh == -1) {
            // All channels used
            freqs = casacore::ArrayColumn<double>(subtbl, "CHAN_FREQ").get(0).tovector();
            chanhigh = freqs.size() - 1;  // Casacore ranges are inclusive

        } else {
            casacore::Slicer slice {
                    casacore::IPosition {chanlow}, casacore::IPosition {chanhigh},
                    casacore::Slicer::endIsLast
            };
            freqs = casacore::ArrayColumn<double>(subtbl, "CHAN_FREQ")
                .getSlice(0, slice)
                .tovector();
        }

        fmt::println("Measurement set {} opened", fname);
        fmt::println(
            "    Channels {} - {} ({:.1f} - {:.1f} MHz) selected",
            chanlow, chanhigh, freqs.front() / 1e6, freqs.back() / 1e6
        );
    }

    auto begin() { return Iterator(*this); }
    auto end() { return Iterator(*this, -1); }

private:
    casacore::Table tbl;
    std::vector<double> freqs;
    int chanlow;
    int chanhigh;
};