#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "aterms.h"
#include "datatable.h"
#include "logger.h"
#include "timer.h"
#include "workunit.h"

std::vector<WorkUnit> partition(DataTable& tbl, GridConfig gridconf, const Aterms& aterms) {
    auto timer = Timer::get("partition");

    // WorkUnit candidate
    // We use candidates as an adhoc structure to store WorkUnits-to-be, before we
    // know their exact position or data range.
    struct WorkUnitCandidate {
        double ulowpx {}, uhighpx, vlowpx, vhighpx;  // in pixels
        double maxspanpx;
        double w0, wmax;                             // in wavelengths
        size_t rowstart;
        size_t chanstart {}, chanend {};
        Aterms::aterm_t aleft, aright;

        WorkUnitCandidate(
            double upx, double vpx, double w0, double maxspanpx, double wmax,
            size_t row, size_t chan, Aterms::aterm_t aleft, Aterms::aterm_t aright
        ) : ulowpx(std::floor(upx)), uhighpx(std::ceil(upx)),
            vlowpx(std::floor(vpx)), vhighpx(std::ceil(vpx)),
            maxspanpx(maxspanpx), w0(w0), wmax(wmax),
            rowstart(row), chanstart(chan), chanend(chan), aleft(aleft), aright(aright) {}

        bool add(
            double upx, double vpx, double w,
            Aterms::aterm_t aleft, Aterms::aterm_t aright, size_t chan
        ) {
            if (add(upx, vpx, w, aleft, aright)) {
                chanend = std::max(chanend, chan);
                return true;
            }

            return false;
        }

        bool add(
            double upx, double vpx, double w,
            Aterms::aterm_t aleft, Aterms::aterm_t aright
        ) {
            // Ensure identical aterm is used
            if (aleft != this->aleft || aright != this->aright) return false;

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

    auto setpositivew = [] (std::array<double, 3> uvw) -> std::array<double, 3> {
        auto& [u, v, w] = uvw;
        if (w >= 0) return uvw;
        return {-u, -v, -w};
    };

    // Initialize workunits which we will return on completion
    std::vector<WorkUnit> workunits;

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

    // Set up candidate workunits. This is mutable state used during the table
    // scan loop below, and will be periodically flushed and reinitialized
    // as WorkUnits are created.
    std::vector<WorkUnitCandidate> candidates;

    // Set up other mutable state used during the loop
    DataTable::Baseline currentbaseline;
    bool createnew {true};

    const auto lambdas = tbl.lambdas();

    for (size_t irow {}; irow < tbl.nrows(); ++irow) {
        // Set current baseline
        currentbaseline = tbl.metadata(irow).baseline;

        // Get left and right Aterms for this time and baseline pair
        auto aleft = aterms.get(tbl.metadata(irow).time, currentbaseline.a);
        auto aright = aterms.get(tbl.metadata(irow).time, currentbaseline.b);

        // For each candidate, check that the first and last channel fit
        for (auto& candidate : candidates) {
            auto [u1, v1, w1] = setpositivew(tbl.uvw(irow, candidate.chanstart));
            auto [u1px, v1px] = padded.UVtoGrid(u1, v1);

            auto [u2, v2, w2] = setpositivew(tbl.uvw(irow, candidate.chanend));
            auto [u2px, v2px] = padded.UVtoGrid(u2, v2);

            if (
                !candidate.add(u1px, v1px, w1, aleft, aright) ||
                !candidate.add(u2px, v2px, w2, aleft, aright)
            ) {
                createnew = true;
                break;
            }
        }

        // Split the channel width into chunks that fit comfortably
        // within a subgrid (and then a little bit extra). Always start new workunits
        // if either the baselines change or we cross into a new timestep.
        if (createnew) {
            // Save any existing candidates as workunits
            for (auto& c : candidates) {
                long long upx = std::llround(c.ulowpx + c.uhighpx) / 2;
                long long vpx = std::llround(c.vlowpx + c.vhighpx) / 2;
                auto [u, v] = padded.gridToUV<double>(upx, vpx);

                double meantime = [&] {
                    double meantime {};
                    for (auto m : tbl.metadata({c.rowstart, irow})) {
                        meantime += m.time;
                    }
                    return meantime / (irow - c.rowstart);
                }();
                double meanfreq = [&] {
                    double meanfreq {};
                    for (auto freq : tbl.freqs({c.chanstart, c.chanend + 1})) {
                        meanfreq += freq;
                    }
                    return meanfreq / (c.chanend + 1 - c.chanstart);
                }();

                workunits.push_back(WorkUnit(
                    meantime, meanfreq, currentbaseline,
                    upx, vpx, u, v, c.w0,
                    c.rowstart, irow, c.chanstart, c.chanend + 1
                ));
            }

            // Reset loop state
            createnew = false;
            candidates.clear();

            for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
                auto [u, v, w] = setpositivew(tbl.uvw(irow, ichan));
                auto [upx, vpx] = padded.UVtoGrid(u, v);

                if (candidates.empty() || !candidates.back().add(upx, vpx, w, aleft, aright, ichan)) {
                    // Create new candidate
                    // We initialize with 0.7 of the full span. This is fudge factor
                    // to allow the uvw positions to move in time, and not immediately
                    // fall outside the subgrid. If this value is too small, we'll
                    // have too many workunits along the frequency axis; if it's too
                    // large, we risk having too many workgrids along the time axis.

                    // Snap w to a w-layer
                    w = (std::floor(w / wstep) + 0.5) * wstep;

                    candidates.push_back(WorkUnitCandidate(
                        upx, vpx, w, 0.8 * maxspanpx, wstep / 2, irow, ichan, aleft, aright
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
        auto [u, v] = padded.gridToUV<double>(upx, vpx);

        double meantime = [&] {
            double meantime {};
            for (auto m : tbl.metadata({c.rowstart, tbl.nrows()})) {
                meantime += m.time;
            }
            return meantime / (tbl.nrows() - c.rowstart);
        }();
        double meanfreq = [&] {
            double meanfreq {};
            for (auto freq : tbl.freqs({c.chanstart, c.chanend + 1})) {
                meanfreq += freq;
            }
            return meanfreq / (c.chanend + 1 - c.chanstart);
        }();

        workunits.push_back(WorkUnit(
            meantime, meanfreq, currentbaseline,
            upx, vpx, u, v, c.w0,
            c.rowstart, tbl.nrows(), c.chanstart, c.chanend + 1
        ));
    }

    // Log WorkUnit statistics
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
            "WorkUnits created: {} Mean occupancy: {:.1f} Occupancy quartiles: {}/{}/{}",
            workunits.size(), mean, occupancy[occupancy.size() / 4],
            occupancy[occupancy.size() / 2], occupancy[3 * occupancy.size() / 4]
        );
    }

    return workunits;
}