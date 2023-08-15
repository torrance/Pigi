#pragma once

#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"
#include "uvdatum.h"

template <typename T>
struct UVWOrigin {
    T u0, v0, w0;

    UVWOrigin(T* ptr) : u0(ptr[0]), v0(ptr[1]), w0(ptr[2]) {}
    UVWOrigin(T u0, T v0, T w0) : u0(u0), v0(v0), w0(w0) {}
};

template <typename S, typename T=HostArray<UVDatum<S>, 1>>
struct WorkUnit {
    long long u0px;
    long long v0px;
    S u0;
    S v0;
    S w0;
    GridSpec subgridspec;
    HostSpan< ComplexLinearData<S>, 2 > Aleft;
    HostSpan< ComplexLinearData<S>, 2 > Aright;
    T data;
};

template <typename T, typename S>
auto partition(
    T uvdata,
    GridSpec gridspec,
    GridSpec subgridspec,
    int padding,
    int wstep,
    HostArray<ComplexLinearData<S>, 2>& Aterms
) {
    // Temporarily store workunits in a map to reduce search space duirng partitioning
    std::unordered_map<
    S, std::vector<WorkUnit< S, std::vector<UVDatum<S>> >>
    > wlayers;
    long long radius {static_cast<long long>(subgridspec.Nx) / 2 - padding};

    for (UVDatum<S> uvdatum : uvdata) {
        // Find equivalent pixel coordinates of (u,v) position
        const auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);

        // Snap to center of nearest wstep window. Note: for a given wstep,
        // the boundaries occur at {0 <= w < wstep, wstep <= w < 2 * wstep, ...}
        // This choice is made because w values are made positive and we want to
        // avoid the first interval including negative values. w0 is the midpoint of each
        // respective interval.
        const S w0 {wstep * std::floor(uvdatum.w / wstep) + static_cast<S>(0.5) * wstep};

        // Search through existing workunits to see if our UVDatum is included in an
        // existing workunit.
        std::vector<WorkUnit< S, std::vector<UVDatum<S>> >>& wworkunits = wlayers[w0];

        bool found {false};
        for (auto& workunit : wworkunits) {
            // The +0.5 accounts for the off-center central pixel of an even grid
            // TODO: add one to upper bound
            if (
                -radius <= upx - workunit.u0px + 0.5 &&
                upx - workunit.u0px + 0.5 <= radius &&
                -radius <= vpx - workunit.v0px + 0.5 &&
                vpx - workunit.v0px + 0.5 <= radius
            ) {
                workunit.data.push_back(uvdatum);
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
            u0px, v0px, u0, v0, w0, subgridspec,
            Aterms, Aterms, std::vector<UVDatum<S>> {uvdatum}
        });
    }

    // Flatten the workunits into a single vector
    // Also swap out the data storage from vector to 1D HostArray
    // since this is a pinned allocation and makes D->H data transfers
    // in the gridder much faster.
    std::vector<WorkUnit< S, HostArray<UVDatum<S>, 1> >> workunits;
    for (auto& [_, wworkunits] : wlayers) {
        while (!wworkunits.empty()) {
            auto& workunit = wworkunits.back();
            // TODO: use emplace_back() when we can upgrade Clang
            workunits.push_back({
                workunit.u0px, workunit.v0px,
                workunit.u0, workunit.v0, workunit.w0,
                workunit.subgridspec, workunit.Aleft, workunit.Aright,
                HostArray<UVDatum<S>, 1>::fromVector(workunit.data)
            });
            wworkunits.pop_back();
        }
    }
    return workunits;
}