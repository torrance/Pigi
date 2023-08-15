#pragma once

#include <string>

#include <casacore/tables/Tables.h>

#include "constants.h"
#include "generator/generator.hpp"
#include "uvdatum.h"

class MeasurementSet {
public:
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

    std::generator<UVDatum<double>> uvdata() {
        std::vector<double> lambdas = freqs;
        for (auto& x : lambdas) { x = Constants::c / x; };  // Convert to lambdas (m)

        casacore::ArrayColumn<double> uvwCol(tbl, "UVW");
        casacore::ScalarColumn<bool> flagrowCol(tbl, "FLAG_ROW");
        casacore::ArrayColumn<bool> flagCol(tbl, "FLAG");
        casacore::ArrayColumn<float> weightCol(tbl, "WEIGHT");
        casacore::ArrayColumn<float> weightspectrumCol(tbl, "WEIGHT_SPECTRUM");
        casacore::ArrayColumn<std::complex<float>> dataCol(tbl, "CORRECTED_DATA");

        // Ensure our expectations about the size of cells are valid.
        // Cells selected using the slice are guaranteed to be correct and we don't need to
        // check again here.
        if (uvwCol.shape(0) != casacore::IPosition{3}) abort();
        if (weightCol.shape(0) != casacore::IPosition {4}) abort();

        casacore::Array<double> uvw;
        casacore::Array<bool> flag;
        casacore::Array<float> weight;
        casacore::Array<float> weightspectrum;
        casacore::Array<std::complex<float>> data;

        for (size_t nrow {}; nrow < tbl.nrow(); ++nrow) {
            // Set up slicers
            casacore::Slicer slice {
                    casacore::IPosition {0, chanlow}, casacore::IPosition {3, chanhigh},
                    casacore::Slicer::endIsLast
            };

            // Fetch row data
            uvwCol.get(nrow, uvw);
            weightCol.get(nrow, weight);
            flagCol.getSlice(nrow, slice, flag);
            weightspectrumCol.getSlice(nrow, slice, weightspectrum);
            dataCol.getSlice(nrow, slice, data);

            auto uvwIter = uvw.begin();
            double u_m {*(uvwIter++)}, v_m {*(uvwIter++)}, w_m {*(uvwIter++)};

            bool flagrow {flagrowCol.get(nrow)};

            LinearData<double> weightRow;
            auto weightIter = weight.begin();
            weightRow.xx = *weightIter; ++weightIter;
            weightRow.xy = *weightIter; ++weightIter;
            weightRow.yx = *weightIter; ++weightIter;
            weightRow.yy = *weightIter; ++weightIter;
            weightRow *= !flagrow;  // Flagged row has the effect to set all to zero

            auto dataIter = data.begin();
            auto weightspectrumIter = weightspectrum.begin();
            auto flagIter = flag.begin();

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
                    data.adjoint();
                    weights.adjoint();
                }

                co_yield UVDatum<double> {
                    nrow, ncol, u, v, w, weights, data
                };
            }
        }
    }

private:
    casacore::Table tbl;
    std::vector<double> freqs;
    int chanlow;
    int chanhigh;
};