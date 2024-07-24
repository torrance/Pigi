#pragma once

#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "aterms.h"
#include "gridspec.h"
#include "logger.h"
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

/**
 * PartitionKey is used as a key in a std::unordered_map during partition to reduce
 * the search sapce. We define it here so that we can make it hashable.
 */
template <typename S>
using PartitionKey = std::tuple<
    double,
    std::shared_ptr<HostArray<ComplexLinearData<S>, 2>>,
    std::shared_ptr<HostArray<ComplexLinearData<S>, 2>>
>;

template <typename S>
struct PartitionKeyHasher {
    std::size_t operator()(const PartitionKey<S>& key) const {
        size_t seed {};
        boost::hash_combine(seed, std::get<0>(key));
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        return seed;
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

    // wstep is the distance between w-layers. We want to choose as large a value as
    // possible to reduce the number of workunits, but we must ensure that |w - w0|
    // remains small enough that the subgrid can properly sample the w-term.
    //
    // We use a simple heuristic that seems to work (i.e. the tests pass).
    // The w-component of the measurement equation, exp(2pi i w n') has a 'variable'
    // frequency, since n' is a function of (l, m). So we take the derivative w.r.t.
    // l and m to find the 'instantaneous' frequency, and substitute in values that give
    // the maximum value. We ensure that, in the worst case, each radiating fringe pattern
    // is sampled at least 3 times.
    double wstep {std::numeric_limits<double>::max()};
    {
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
            wstep = std::min(wstep, maxn / (6 * std::abs(maxl) * subgridspec.scalel));
            wstep = std::min(wstep, maxn / (6 * std::abs(maxm) * subgridspec.scalem));
        }
    }
    Logger::debug("Setting wstep = {}", wstep);

    // Initialize workunits vector
    std::vector<WorkUnit<S>> workunits;

    // Set up some helpful types for using boost::geometry
    using Point = boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
    using Box = boost::geometry::model::box<Point>;
    using Value = std::pair<Box, size_t>;
    using RTree = boost::geometry::index::rtree<
        Value, boost::geometry::index::quadratic<16>
    >;

    // Preallocate rtree_result
    std::vector<Value> rtree_result;

    // Store rtrees in a map, indexed by wstep and A-terms. This reduces
    // the size of each rtree and avoids manually filtering workunits with
    // incompatible A-terms
    std::unordered_map<PartitionKey<S>, RTree, PartitionKeyHasher<S>> rtrees;

    // Calculate the radius of the subgrid, excluding the pixels reserved for
    // padding.
    long long radius {gridconf.kernelsize / 2 - gridconf.kernelpadding};

    for (auto& uvdatum : uvdata) {
        // Find the Aterms for this uvdatum
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
            wstep * std::floor(uvdatum.w / wstep)
            + 0.5 * wstep
        };

        // Get the associated rtree for this combination of w0 and Aterms
        auto& rtree = rtrees[{w0, Aleft, Aright}];

        // Check if (upx, vpx) is already within an existing box
        rtree_result.clear();
        rtree.query(
            boost::geometry::index::contains(Point(upx, vpx)),
            std::back_inserter(rtree_result)
        );

        if (rtree_result.empty()) {
            // No box was found. Let's create a new workunit for our UVDatum
            // First, snap upx, vpx to the nearest pixels
            const long long u0px {llround(upx)}, v0px {llround(vpx)};
            const auto [u0, v0] = gridspec.gridToUV<S>(u0px, v0px);

            // Create a new workunit with uvdatum as first member
            // TODO: use emplace_back() when we can upgrade Clang
            workunits.push_back({
                u0px, v0px, u0, v0, static_cast<S>(w0),
                Aleft, Aright, {&uvdatum}
            });

            // Now add workgroup's box to the rtree
            rtree.insert({
                Box(
                    Point(u0px - 0.5 - radius, v0px - 0.5 - radius), // bottom left corner
                    Point(u0px - 0.5 + radius, v0px - 0.5 + radius)  // upper right corner
                ),
                workunits.size() - 1  // associated index into workunits
            });
        } else {
            // We found an overlapping box. Add uvdatum to the existing workunit.
            WorkUnit<S>& workunit = workunits[rtree_result.front().second];
            workunit.data.push_back(&uvdatum);
        }
    }

    return workunits;
}