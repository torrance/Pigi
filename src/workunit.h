#pragma once

#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "aterms.h"
#include "gridspec.h"
#include "memmap.h"
#include "memory.h"
#include "outputtypes.h"
#include "uvdatum.h"

template <typename T>
struct UVWOrigin {
    T u0, v0, w0;

    UVWOrigin(T* ptr) : u0(ptr[0]), v0(ptr[1]), w0(ptr[2]) {}
    UVWOrigin(T u0, T v0, T w0) : u0(u0), v0(v0), w0(w0) {}
};

template <typename S>
struct WorkUnit {
    long long u0px;
    long long v0px;
    S u0;
    S v0;
    S w0;
    std::shared_ptr<HostArray< ComplexLinearData<S>, 2 >> Aleft;
    std::shared_ptr<HostArray< ComplexLinearData<S>, 2 >> Aright;
    std::vector<UVDatum<S>*> data;

    template <typename P>
    LinearData<P> totalWeight() const {
        LinearData<double> w {};
        for (const auto uvdatumptr : this->data) {
            w += static_cast<LinearData<double>>(uvdatumptr->weights);
        }
        return static_cast<LinearData<P>>(w);
    }
};

template <typename S>
auto partition(
    auto&& uvdata,
    const GridConfig gridconf,
    const Aterms<S>& aterms
) {
    // We use the padded gridspec during partitioning
    auto gridspec = gridconf.padded();

    // Temporarily store workunits in a map to reduce search space during partitioning
    std::unordered_map<double, std::vector<WorkUnit<S>>> wlayers;
    long long radius {gridconf.kernelsize / 2 - gridconf.kernelpadding};

    for (auto& uvdatum : uvdata) {
        // Find Aterm
        auto Aleft = aterms.get(uvdatum.meta->time, uvdatum.meta->ant1);
        auto Aright = aterms.get(uvdatum.meta->time, uvdatum.meta->ant2);

        // Find equivalent pixel coordinates of (u,v) position
        const auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);

        // Snap to center of nearest wstep window. Note: for a given wstep,
        // the boundaries occur at {0 <= w < wstep, wstep <= w < 2 * wstep, ...}
        // This choice is made because w values are made positive and we want to
        // avoid the first interval including negative values. w0 is the midpoint of each
        // respective interval.
        const double w0 {
            gridconf.wstep * std::floor(uvdatum.w / gridconf.wstep)
            + 0.5 * gridconf.wstep
        };

        // Search through existing workunits to see if our UVDatum is included in an
        // existing workunit.
        auto& wworkunits = wlayers[w0];

        bool found {false};
        for (auto& workunit : wworkunits) {
            // The +0.5 accounts for the off-center central pixel of an even grid
            // TODO: add one to upper bound
            if (
                workunit.Aleft == Aleft &&
                workunit.Aright == Aright &&
                -radius <= upx - workunit.u0px + 0.5 &&
                upx - workunit.u0px + 0.5 <= radius &&
                -radius <= vpx - workunit.v0px + 0.5 &&
                vpx - workunit.v0px + 0.5 <= radius
            ) {
                workunit.data.push_back(&uvdatum);
                found = true;
                break;
            }
        }
        if (found) continue;

        // If we are here, we need to create a new workunit for our UVDatum
        const long long u0px {llround(upx)}, v0px {llround(vpx)};
        const auto [u0, v0] = gridspec.gridToUV<S>(u0px, v0px);

        // TODO: use emplace_back() when we can upgrade Clang
        wworkunits.push_back({
            u0px, v0px, u0, v0, static_cast<S>(w0),
            Aleft, Aright, {&uvdatum}
        });
    }

    // Flatten workunits
    std::vector<WorkUnit<S>> workunits;
    for (auto& [_, wworkunits] : wlayers) {
        for (auto& workunit : wworkunits) {
            workunits.push_back({
                workunit.u0px, workunit.v0px,
                workunit.u0, workunit.v0, workunit.w0,
                workunit.Aleft, workunit.Aright, std::move(workunit.data)
            });
        }
    }

    return workunits;
}