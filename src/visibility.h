#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <utility>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableIter.h>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "outputtypes.h"

class Visibility {
public:
    struct Baseline {
        int a {};
        int b {};

        bool operator==(const Baseline& other) {
            return a == other.a && b == other.b;
        }

        bool operator!=(const Baseline& other) { return !(*this == other); }
    };

    struct RowMetadata {
        double time {};
        Baseline baseline {};
        double u {}, v {}, w{};
    };

    struct Workunit {
        // Location within master grid
        long long upx, vpx;
        double u, v, w;

        // Data slice
        size_t rowstart, rowend;
        size_t chanstart, chanend;
    };

    Visibility(const std::string& path) {
        casacore::MeasurementSet mset(path);
        // TODO: Allow selection of field ID.
        // TODO: Slice based on chanlow, chanhigh

        // Remove autocorrelations
        mset = mset(mset.col("ANTENNA1") != mset.col("ANTENNA2"));

        // Remove flagged rows
        mset = mset(!mset.col("FLAG_ROW"));

        nrow = mset.nrow();

        // Retrieve frequency values
        {
            auto spw = mset.spectralWindow();
            freqs = casacore::ArrayColumn<double>(spw, "CHAN_FREQ").get(0).tovector();

            // TODO: Validate frequencies are strictly ASC

            for (double freq : freqs) lambdas.push_back(Constants::c / freq);
        }

        // fmt::println("Reading table with {} rows and {} channels (after filtering)...", nrow, freqs.size());

        // Allocate data arrays
        metadata.resize(nrow);
        weights.resize(nrow * freqs.size());
        data.resize(nrow * freqs.size());

        casacore::Block<casacore::String> sortedcols(2);
        sortedcols[0] = "ANTENNA1";
        sortedcols[1] = "ANTENNA2";

        size_t irow {};
        for (casacore::TableIterator iter(mset, sortedcols); !iter.pastEnd(); iter.next()) {
            auto subtbl = iter.table();
            size_t nsubrow = subtbl.nrow();

            // Skip processing of an empty table
            if (nsubrow == 0) continue;

            int ant1 = casacore::ScalarColumn<int>(subtbl, "ANTENNA1").get(0);
            int ant2 = casacore::ScalarColumn<int>(subtbl, "ANTENNA2").get(0);

            // Create metadata
            {
                auto timeCol = casacore::ScalarColumn<double>(
                    subtbl, "TIME_CENTROID"
                ).getColumn();
                auto uvwCol = casacore::ArrayColumn<double>(subtbl, "UVW").getColumn();

                auto timeIter = timeCol.begin();
                auto uvwIter = uvwCol.begin();
                for (size_t i {}; i < nsubrow; ++i) {
                    double u = *uvwIter; ++uvwIter;
                    double v = *uvwIter; ++uvwIter;
                    double w = *uvwIter; ++uvwIter;

                    metadata[irow + i] = RowMetadata{
                        .time = *timeIter,
                        .baseline = Baseline{ant1, ant2},
                        .u = u, .v = v, .w = w
                    };

                    ++timeIter;
                }
            }

            // Copy weight spectrum in full
            {
                auto weightSpectrumCol = casacore::ArrayColumn<float>(
                    subtbl, "WEIGHT_SPECTRUM"
                ).getColumn();

                std::copy(
                    weightSpectrumCol.begin(), weightSpectrumCol.end(),
                    reinterpret_cast<float*>(weights.data() + irow * freqs.size())
                );
            }

            // Copy across the data
            {
                auto dataCol = casacore::ArrayColumn<std::complex<float>>(
                    subtbl, "CORRECTED_DATA"
                ).getColumn();

                std::copy(
                    dataCol.begin(), dataCol.end(),
                    reinterpret_cast<std::complex<float>*>(data.data() + irow * freqs.size())
                );
            }

            // Then fold flag and NaNs from the data column into weights
            {
                auto flagCol = casacore::ArrayColumn<bool>(subtbl, "FLAG").getColumn();
                auto flagIter = flagCol.begin();

                for (size_t i {}; i < nsubrow; ++i) {
                    for (size_t j {}; j < freqs.size(); ++j) {
                        auto& datum = data[(irow + i) * freqs.size() + j];
                        auto& weight = weights[(irow + i) * freqs.size() + j];

                        // Treat NaNs as flags
                        if (!datum.isfinite() || !weight.isfinite()) {
                            datum = {0, 0, 0, 0};
                            weight = {0, 0, 0, 0};
                        }

                        // Fold flag values into the weight column
                        weight.xx *= !(*flagIter); ++flagIter;
                        weight.xy *= !(*flagIter); ++flagIter;
                        weight.yx *= !(*flagIter); ++flagIter;
                        weight.yy *= !(*flagIter); ++flagIter;
                    }
                }
            }

            irow += nsubrow;
        }

        // // Collapse

        // // Set all uvw to be positive
        // // Maybe perform this in the gridder kernel?
        // for (size_t i {}; i < nrow; ++i) {
        //     auto& m = metadata[i];
        //     if (m.w < 0) {
        //         m.u *= -1;
        //         m.v *= -1;
        //         m.w *= -1;

        //         for (size_t j {}; j < freqs.size(); ++j) {
        //             assert(data[i * freqs.size() + j].isfinite());
        //             assert(weights[i * freqs.size() + j].isfinite());
        //             data[i * freqs.size() + j] = data[i * freqs.size() + j].adjoint();
        //         }
        //     }
        // }
    }

    size_t mem() {
        return (
            freqs.size() * sizeof(double) +
            metadata.size() * sizeof(RowMetadata) +
            weights.size() * sizeof(LinearData<float>) +
            data.size() * sizeof(ComplexLinearData<float>)
        );
    }

    std::vector<Workunit> partition(GridConfig gridconf) {
        // Workunit candidate
        struct WorkunitCandidate {
            double ulowpx {}, uhighpx, vlowpx, vhighpx;  // in pixels
            double maxspanpx;
            double w0, wmax;                             // in wavelengths
            size_t chanstart {}, chanend {};
            size_t rowstart;

            WorkunitCandidate(
                double upx, double vpx, double w0, double maxspanpx, double wmax,
                size_t row, size_t chan
            ) : ulowpx(std::floor(upx)), uhighpx(std::ceil(upx)),
                vlowpx(std::floor(vpx)), vhighpx(std::ceil(vpx)),
                maxspanpx(maxspanpx), w0(w0), wmax(wmax),
                rowstart(row), chanstart(chan), chanend(chan) {}

            bool add(double upx, double vpx, double w, size_t chan) {
                if (add(upx, vpx, w)) {
                    chanend = std::max(chanend, chan);
                    return true;
                }

                return false;
            }

            bool add(double upx, double vpx, double w) {
                // Check w is within range
                if (std::abs(w - w0) > wmax) {
                    // fmt::println("Failed w test: {} versus {}", w, w0);
                    return false;
                }

                // Check that this visbilility fits, and if so update the bounds
                double ulowpx_ = std::floor(std::min(upx, ulowpx));
                double uhighpx_ = std::ceil(std::max(upx, uhighpx));
                double vlowpx_ = std::floor(std::min(vpx, vlowpx));
                double vhighpx_ = std::ceil(std::max(vpx, vhighpx));

                if (
                    uhighpx_ - ulowpx_ > maxspanpx ||
                    vhighpx_ - vlowpx_ > maxspanpx
                ) {
                    // fmt::println("Not in span: ({}, {}) not in {}-{} {}-{}", upx, vpx, ulowpx, uhighpx, vlowpx, vhighpx);
                    return false;
                }

                ulowpx = ulowpx_;
                uhighpx = uhighpx_;
                vlowpx = vlowpx_;
                vhighpx = vhighpx_;

                return true;
            }
        };

        std::vector<Workunit> workunits;

        // Get the padded and subgrid GridSpec objections
        GridSpec subgrid = gridconf.subgrid();
        GridSpec padded = gridconf.padded();

        const double wmax = 20;
        const double wstep = 30; // TODO calculate

        const double maxspanpx = subgrid.Nx - 2 * gridconf.kernelpadding;

        // Set up candidate workunits
        std::vector<WorkunitCandidate> candidates;

        // We use priorbaseline to track changes in the baseline during the loop
        std::optional<Baseline> priorbaseline;

        for (size_t irow {}; irow < nrow; ++irow) {
            auto m = metadata[irow];

            // For each candidate, check that the first and last channel fit
            // fmt::println("Testing candidates on new row...");
            for (auto& candidate : candidates) {
                double u1px = m.u / lambdas[candidate.chanstart] / subgrid.scaleu;
                double v1px = m.v / lambdas[candidate.chanstart] / subgrid.scalev;
                double w1 = m.w / lambdas[candidate.chanstart];

                double u2px = m.u / lambdas[candidate.chanend] / subgrid.scaleu;
                double v2px = m.v / lambdas[candidate.chanend] / subgrid.scalev;
                double w2 = m.w / lambdas[candidate.chanend];

                if (!candidate.add(u1px, v1px, w1) || !candidate.add(u2px, v2px, w2)) {
                    priorbaseline.reset();  // use this value as a signal to create new candidates
                    break;
                }
            }
            // fmt::println("Done");

            // Split the channel width into chunks that fit comfortably
            // within a subgrid (and then a little bit extra)
            if (!priorbaseline || m.baseline != priorbaseline.value()) {
                priorbaseline = m.baseline;

                // Save any existing candidates as workunits
                for (auto& c : candidates) {
                    long long upx = std::llround(c.ulowpx + c.uhighpx) / 2;
                    long long vpx = std::llround(c.vlowpx + c.vhighpx) / 2;
                    double u = upx * padded.scaleu;
                    double v = vpx * padded.scalev;

                    // Offset wrt to the bottom left corner
                    upx += padded.Nx / 2;
                    vpx += padded.Ny / 2;

                    workunits.push_back(Workunit(
                        upx, vpx, u, v, c.w0,
                        c.rowstart, irow, c.chanstart, c.chanend + 1
                    ));
                }

                // if (!candidates.empty()) {
                //     auto& c = candidates.back();
                //     fmt::println("Creating workunits with {} rows and {} channel chunks", irow - c.rowstart, candidates.size());
                // }

                candidates.clear();
                for (size_t chan {}; chan < freqs.size(); ++chan) {
                    double upx = m.u / lambdas[chan] / subgrid.scaleu;
                    double vpx = m.v / lambdas[chan] / subgrid.scalev;
                    double w = m.w / lambdas[chan];

                    if (
                        candidates.empty() ||
                        !candidates.back().add(upx, vpx, w, chan)
                    ) {
                        // Create new candidate
                        // We initialize with 0.7 of the full span. This is fudge factor
                        // to allow the uvw positions to move in time, and not immediately
                        // fall outside the subgrid. If this value is too small, we'll
                        // have too many workunits along the frequency axis; if it's too
                        // large, we risk having too many workgrids along the time axis.

                        // Snap w to a w-layer
                        w = (std::floor(w / wstep) + 0.5) * wstep;

                        candidates.push_back(WorkunitCandidate(
                            upx, vpx, w, 0.7 * maxspanpx, wstep / 2, irow, chan
                        ));
                    }
                }

                // Set all candidates to use the full span
                for (auto& candidate : candidates) {
                    candidate.maxspanpx = maxspanpx;
                    candidate.wmax = wmax;
                }

                // fmt::println("Created new candidates: ");
                // for (auto& c : candidates) {
                //     fmt::print("{}-{} ", c.chanstart, c.chanend);
                // }
                // fmt::println("");
            }
        }

        // Save final candidates as workunits
        for (auto& c : candidates) {
            long long upx = std::llround(c.ulowpx + c.uhighpx) / 2;
            long long vpx = std::llround(c.vlowpx + c.vhighpx) / 2;
            double u = upx * padded.scaleu;
            double v = vpx * padded.scalev;

            // Offset wrt to the bottom left corner
            upx += padded.Nx / 2;
            vpx += padded.Ny / 2;

            workunits.push_back(Workunit(
                upx, vpx, u, v, c.w0,
                c.rowstart, nrow, c.chanstart, c.chanend + 1
            ));
        }

        return workunits;
    }

    // TODO: Investigate storing baselines as std::unorded_map?
    size_t nrow {};
    std::vector<double> freqs;
    std::vector<double> lambdas;
    std::vector<RowMetadata> metadata;
    std::vector<LinearData<float>> weights;
    std::vector<ComplexLinearData<float>> data;
};