#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>

#include "gridspec.h"
#include "invert.h"
#include "mset.h"
#include "memory.h"
#include "outputtypes.h"
#include "predict.h"
#include "taper.h"
#include "uvdatum.h"
#include "workunit.h"

template <typename F>
auto simple_benchmark(std::string_view name, const int N, const F f) {
    // Perform one warm up and keep the result to return
    auto ret = f();

    std::vector<double> timings;
    for (int n {}; n < N; ++n) {
        auto start = std::chrono::steady_clock::now();
        f();
        auto end = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        );
    }

    // Calculate mean, median and variance of timings
    std::sort(timings.begin(), timings.end());
    double median {timings[N / 2]};
    if (N % 2 == 1) {
        // Special case for even samples
        median += timings[N / 2 -1];
        median /= 2;
    }
    auto mean = std::accumulate(timings.begin(), timings.end(), 0.) / N;
    auto variance = std::transform_reduce(
        timings.begin(), timings.end(), 0., std::plus{}, [=] (auto timing) {
            return std::pow(timing - mean, 2);
        }
    );

    fmt::println(
        "Benchmark: {} ({} samples) mean: {:.3f} +/- {:.3f} s median: {:.3f} s",
        name, N, mean / 1000, std::sqrt(variance) / 1000, median / 1000
    );

    return ret;
}

TEST_CASE("MSet reading and paritioning", "[io]") {
    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    HostArray<ComplexLinearData<double>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        "/home/torrance/testdata/1215555160/1215555160.ms",
        {.chanlow = 0, .chanhigh = 191}
    );

    auto uvdata = simple_benchmark("MSet read", 5, [&] {
        std::vector<UVDatum<double>> uvdata;
        for (auto& uvdatum : mset) {
            uvdata.push_back(uvdatum);
        }
        return uvdata;
    });

    auto workunits = simple_benchmark("Partition", 5, [&] {
        return partition(uvdata, gridspec, subgridspec, 18, 25, Aterms);
    });
}

TEMPLATE_TEST_CASE("Invert", "[invert]", float, double) {
    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);

    HostArray<ComplexLinearData<TestType>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        "/home/torrance/testdata/1215555160/1215555160.ms",
        {.chanlow = 0, .chanhigh = 11}
    );

    // Convert to TestType precision
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : mset) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    auto workunits = partition(uvdata, gridspec, subgridspec, 18, 25, Aterms);

    simple_benchmark("Invert", 5, [&] {
        return invert<StokesI, TestType>(
            workunits, gridspec, taper, subtaper
        );
    });
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);

    HostArray<ComplexLinearData<TestType>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        "/home/torrance/testdata/1215555160/1215555160.ms",
        {.chanlow = 0, .chanhigh = 11}
    );

    // Convert to TestType precision
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : mset) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    auto workunits = partition(uvdata, gridspec, subgridspec, 18, 25, Aterms);

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap({gridspec.Nx, gridspec.Ny});

    simple_benchmark("Predict", 5, [&] {
        predict<StokesI<TestType>, TestType>(
            workunits, skymap, gridspec, taper, subtaper, DegridOp::Replace
        );
        return true;
    });
}