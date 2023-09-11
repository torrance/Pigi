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
        Iterator(MeasurementSet& mset, size_t nrow = 0) :
            mset(mset), lambdas(mset.freqs), nrow(nrow - 1),
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
            return nrow == other.nrow;
        }

        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        MeasurementSet& mset;
        std::vector<double> lambdas;
        std::vector<UVDatum<double>> cache;
        std::vector<UVDatum<double>>::iterator cache_current;
        size_t nrow {};

        casacore::ArrayColumn<double> uvwCol;
        casacore::ScalarColumn<bool> flagrowCol;
        casacore::ArrayColumn<bool> flagCol;
        casacore::ArrayColumn<float> weightCol;
        casacore::ArrayColumn<float> weightspectrumCol;
        casacore::ArrayColumn<std::complex<float>> dataCol;

        casacore::Array<double> uvwArray;
        casacore::Array<bool> flagArray;
        casacore::Array<float> weightArray;
        casacore::Array<float> weightspectrumArray;
        casacore::Array<std::complex<float>> dataArray;

        void rebuild_cache() {
            // Reset the cache and cache pointer
            // Do this always, as end() relies on this behaviour
            cache.clear();
            cache_current = cache.begin();

            // Early return if we've finished the table
            if (++nrow >= mset.tbl.nrow()) return;

            // Set up slicers
            casacore::Slicer slice {
                    casacore::IPosition {0, mset.chanlow}, casacore::IPosition {3, mset.chanhigh},
                    casacore::Slicer::endIsLast
            };

            // Fetch row data in arrays
            uvwCol.get(nrow, uvwArray);
            weightCol.get(nrow, weightArray);
            flagCol.getSlice(nrow, slice, flagArray);
            weightspectrumCol.getSlice(nrow, slice, weightspectrumArray);
            dataCol.getSlice(nrow, slice, dataArray);

            auto uvwIter = uvwArray.begin();
            double u_m {*(uvwIter++)}, v_m {*(uvwIter++)}, w_m {*(uvwIter++)};

            bool flagrow {flagrowCol.get(nrow)};

            LinearData<double> weightRow;
            auto weightIter = weightArray.begin();
            weightRow.xx = *weightIter; ++weightIter;
            weightRow.xy = *weightIter; ++weightIter;
            weightRow.yx = *weightIter; ++weightIter;
            weightRow.yy = *weightIter; ++weightIter;
            weightRow *= !flagrow;  // Flagged row has the effect to set all to zero

            auto dataIter = dataArray.begin();
            auto weightspectrumIter = weightspectrumArray.begin();
            auto flagIter = flagArray.begin();

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
                    nrow, ncol, u, v, w, weights, data
                });
            }

            cache_current = cache.begin(); // Reset iterator, as it may have changed
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
    auto end() { return Iterator(*this, tbl.nrow()); }

private:
    casacore::Table tbl;
    std::vector<double> freqs;
    int chanlow;
    int chanhigh;
};