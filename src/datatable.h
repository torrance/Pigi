#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

#include <casacore/casa/Utilities/Compare.h>
#include <casacore/casa/Utilities/CountedPtr.h>
#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableIter.h>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "logger.h"
#include "outputtypes.h"

class DataTable {
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

    DataTable(
        const std::string& path,
        double maxduration,
        long long chanlow,
        long long chanhigh
    ) {
        casacore::MeasurementSet mset(path);

        // TODO: Allow selection of field ID.

        // Remove autocorrelations
        mset = mset(mset.col("ANTENNA1") != mset.col("ANTENNA2"));

        // Remove flagged rows
        mset = mset(!mset.col("FLAG_ROW"));

        // Retrieve frequency values
        {
            auto spw = mset.spectralWindow();
            casacore::ArrayColumn<double> chanCol(spw, "CHAN_FREQ");
            auto allfreqs = chanCol.get(0).tovector();

            // Ensure chanlow >= 0; and set chanhigh to default to all channels if <= 0
            chanlow = std::min(0ll, chanlow);
            if (chanhigh <= 0) chanhigh = allfreqs.size();

            for (long long chan {chanlow}; chan < chanhigh; ++chan) {
                freqs.push_back(allfreqs[chan]);
            }

            for (double freq : freqs) lambdas.push_back(Constants::c / freq);
        }

        // Set dimensions of table
        nrows = mset.nrow();
        nchans = chanhigh - chanlow;

        Logger::info(
            "Reading table with {} rows and {} channels (after filtering)...",
            nrows, nchans
        );

        // Allocate data arrays
        metadata.resize(nrows);
        weights.resize(nrows * nchans);
        data.resize(nrows * nchans);

        // Determine initial time for use in TableIterator
        double initialtime = [&] {
            auto times = casacore::ScalarColumn<double>(mset, "TIME_CENTROID").getColumn();
            return *std::min_element(times.begin(), times.end());
        }();

        if (maxduration <= 0) maxduration = std::numeric_limits<double>::max();

        // We iterate through the measurement set in [time block] x [baselines]
        // Each time block is <= maxduration. This order is essential for
        // efficient partitioning.
        casacore::Block<casacore::String> colnames(3);
        colnames[0] = "TIME_CENTROID";
        colnames[1] = "ANTENNA1";
        colnames[2] = "ANTENNA2";

        casacore::Block<std::shared_ptr<casacore::BaseCompare>> cmpobjs(3);
        auto interval = std::make_shared<casacore::CompareIntervalReal<double>>(
            initialtime, maxduration
        );
        cmpobjs[0] = interval;
        cmpobjs[1] = std::shared_ptr<casacore::BaseCompare>();
        cmpobjs[2] = std::shared_ptr<casacore::BaseCompare>();

        casacore::Block<casacore::Int> orders(3);
        orders[0] = 0;
        orders[1] = 0;
        orders[2] = 0;

        size_t irow {};
        for (
            casacore::TableIterator iter(mset, colnames, cmpobjs, orders);
            !iter.pastEnd();
            iter.next()
        ) {
            auto subtbl = iter.table();
            size_t nsubrows = subtbl.nrow();

            // Skip processing of an empty table
            if (nsubrows == 0) continue;

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
                for (size_t i {}; i < nsubrows; ++i) {
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
                    reinterpret_cast<float*>(weights.data() + irow * nchans)
                );
            }

            // Copy across the data
            {
                auto dataCol = casacore::ArrayColumn<std::complex<float>>(
                    subtbl, "CORRECTED_DATA"
                ).getColumn();

                std::copy(
                    dataCol.begin(), dataCol.end(),
                    reinterpret_cast<std::complex<float>*>(data.data() + irow * nchans)
                );
            }

            // Then fold flag and NaNs from the data column into weights
            {
                auto flagCol = casacore::ArrayColumn<bool>(subtbl, "FLAG").getColumn();
                auto flagIter = flagCol.begin();

                for (size_t i {}; i < nsubrows; ++i) {
                    for (size_t j {}; j < nchans; ++j) {
                        auto& datum = data[(irow + i) * nchans + j];
                        auto& weight = weights[(irow + i) * nchans + j];

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

            irow += nsubrows;
        }
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
        // We use candidates as an adhoc structure to store Workunits-to-be, before we
        // know their exact position or data range.
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

        // Initialize workunits which we will return on completion
        std::vector<Workunit> workunits;

        // Get the padded and subgrid GridSpec objections
        GridSpec subgrid = gridconf.subgrid();
        GridSpec padded = gridconf.padded();

        // wstep is the maximum distance up until a visibility is critically sampled over
        // a subgrid. We want to choose as large a value as possible to reduce the number
        // of workunits, but we must ensure that |w - w0| remains small enough that the
        // subgrid can properly sample the w-term.
        //
        // We use a simple heuristic that seems to work (i.e. the tests pass).
        // The w-component of the measurement equation, exp(2pi i w n') has a 'variable'
        // frequency, since n' is a function of (l, m). So we take the derivative w.r.t.
        // l and m to find the 'instantaneous' frequency, and substitute in values that give
        // the maximum value. We ensure that, in the worst case, each radiating fringe pattern
        // is sampled at least 3 times.
        const double wmax = [&] {
            double wmax {std::numeric_limits<double>::max()};
            auto subgridspec = gridconf.subgrid();

            std::array<size_t, 4> corners {
                0,  // bottom left
                static_cast<size_t>(subgridspec.Nx) - 1,  // bottom right
                subgridspec.size() - static_cast<size_t>(subgridspec.Nx),  // top left
                subgridspec.size() - 1  // top right
            };

            for (size_t i : corners) {
                auto [maxl, maxm] = subgridspec.linearToSky<double>(i);
                auto maxn = std::sqrt(1 - maxl * maxl - maxm * maxm);

                // Consider sampling density in both dl, and dm directions
                wmax = std::min(wmax, maxn / (12 * std::abs(maxl) * subgridspec.scalel));
                wmax = std::min(wmax, maxn / (12 * std::abs(maxm) * subgridspec.scalem));
            }

            return wmax;
        }();
        Logger::debug("Calculated wmax = {}", wmax);

        // wstep controls the position of the wlayers. Set this at 80% of the allowable
        // limit to allow partitioning to have some space to grow before triggering a new
        // workunit.
        const double wstep = std::llround(2 * wmax * 0.8);
        Logger::debug("Setting wstep = {}", wstep);

        // Maxspanpx is the maximum allowable span in either the u or v direction across
        // a subgrid. Note: it assumes a square subgrid.
        const double maxspanpx = subgrid.Nx - 2 * gridconf.kernelpadding;

        // Set up candidate workunits
        std::vector<WorkunitCandidate> candidates;

        // We use priorbaseline to track changes in the baseline during the loop
        std::optional<Baseline> priorbaseline;

        for (size_t irow {}; irow < nrows; ++irow) {
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

                candidates.clear();
                for (size_t chan {}; chan < nchans; ++chan) {
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
                c.rowstart, nrows, c.chanstart, c.chanend + 1
            ));
        }

        // Log Workunit statistics
        {
            std::vector<int> occupancy;
            for (auto& workunit : workunits) {
                occupancy.push_back(
                    (workunit.chanend - workunit.chanstart) * (workunit.rowend - workunit.rowstart)
                );
            }
            std::sort(occupancy.begin(), occupancy.end());

            double mean = std::accumulate(
                occupancy.begin(), occupancy.end(), 0.
            ) / occupancy.size();

            Logger::debug(
                "Workunits created: {} Mean occupancy: {:.1f} Occupancy quartiles: {}/{}/{}",
                workunits.size(), mean, occupancy[occupancy.size() / 4],
                occupancy[occupancy.size() / 2], occupancy[3 * occupancy.size() / 4]
            );
        }

        return workunits;
    }

    // TODO: Investigate storing baselines as std::unorded_map?
    size_t nrows {};
    size_t nchans {};
    std::vector<double> freqs;
    std::vector<double> lambdas;
    std::vector<RowMetadata> metadata;
    std::vector<LinearData<float>> weights;
    std::vector<ComplexLinearData<float>> data;
};