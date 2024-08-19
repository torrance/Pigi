#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>

#define PIGI_TIMER_ENABLE = 1

#include "aterms.h"
#include "clean.h"
#include "datatable.h"
#include "gridspec.h"
#include "invert.h"
#include "memory.h"
#include "outputtypes.h"
#include "predict.h"
#include "taper.h"
#include "workunit.h"
#include "weighter.h"

const std::vector<casacore::MeasurementSet> TESTDATA = [] {
    // TESTDATA may include multiple, space-separated paths
    std::string paths = getenv("TESTDATA");
    fmt::println("{}", paths);

    std::vector<casacore::MeasurementSet> msets;
    for (size_t i0 {}, i1 {}; i1 <= paths.size(); ++i1) {
        if (i1 == paths.size() || paths[i1] == ' ') {
            msets.push_back({paths.substr(i0, i1)});
            i0 = ++i1;
        }
    }

    return msets;
}();

template <typename F>
auto simple_benchmark(std::string_view name, const int N, const F f) {
    decltype(f()) ret;

    std::vector<double> timings;
    for (int n {}; n < N; ++n) {
        auto start = std::chrono::steady_clock::now();
        ret = f();
        auto end = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        );
    }

    // Calculate mean, median and variance of timings
    std::sort(timings.begin(), timings.end());
    double median {timings[N / 2]};
    if (N % 2 == 1 && N > 1) {
        // Special case for even samples
        median += timings[N / 2 -1];
        median /= 2;
    }
    auto fastest = *std::min_element(timings.begin(), timings.end());
    auto mean = std::accumulate(timings.begin(), timings.end(), 0.) / N;
    double variance {};
    for (auto timing : timings) {
        variance += std::pow(timing - mean, 2) / N;
    }

    fmt::println(
        "Benchmark: {} ({} samples) mean: {:.6f} +/- {:.6f} s median: {:.6f} s min: {:.6f} s",
        name, N, mean / 1000000, std::sqrt(variance) / 1000000, median / 1000000, fastest / 1000000
    );

    return ret;
}

TEST_CASE("MSet reading and partitioning", "[io]") {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .kernelsize = 96, .kernelpadding = 18
    };

    auto tbl = simple_benchmark("MSet read", 1, [&] {
        return DataTable(TESTDATA, {.chanlow=0, .chanhigh=192});
    });

    auto workunits = simple_benchmark("Partition", 5, [&] {
        return partition(tbl, gridconf, 30);
    });
}

TEMPLATE_TEST_CASE("Invert", "[invert]", float, double) {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .paddingfactor=1, .kernelsize = 96, .kernelpadding = 18
    };

    auto beam = Beam::Uniform<double>().gridResponse(
        gridconf.subgrid(), {0, 0}, 0)
    ;
    Aterms aterms(beam);

    DataTable tbl(TESTDATA, {.chanlow=0, .chanhigh=384});
    auto workunits = partition(tbl, gridconf);

    // Prefill any caches, e.g. taper
    invert<StokesI, TestType>(tbl, workunits, gridconf, aterms);

    simple_benchmark("Invert", 5, [&] {
        return invert<StokesI, TestType>(
            tbl, workunits, gridconf, aterms
        );
    });
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .kernelsize = 96, .kernelpadding = 18
    };

    auto beam = Beam::Uniform<double>().gridResponse(
        gridconf.subgrid(), {0, 0}, 0
    );
    Aterms aterms(beam);

    DataTable tbl(TESTDATA, {.chanlow=0, .chanhigh=384});
    auto workunits = partition(tbl, gridconf);

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap {gridconf.grid().shape()};

    // Prefill any caches, e.g. taper
    predict<StokesI, TestType>(
        tbl, workunits, skymap, gridconf, aterms, DegridOp::Add
    );

    simple_benchmark("Predict", 5, [&] {
        predict<StokesI, TestType>(
            tbl, workunits, skymap, gridconf, aterms, DegridOp::Add
        );
        return true;
    });
}

TEST_CASE("Image size", "[imagesize]") {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    DataTable tbl(TESTDATA, {.chanlow=0, .chanhigh=384});

    for (int i : {1, 2, 4, 8, 12, 16, 24, 32, 40, 48}) {
        GridConfig gridconf {
            .imgNx = 1000 * i, .imgNy = 1000 * i, .imgScalelm = std::sin(deg2rad(15. / 3600 / i)),
            .paddingfactor=1, .kernelsize = 96, .kernelpadding = 18
        };

        auto beam = Beam::Uniform<double>().gridResponse(gridconf.subgrid(), {0, 0}, 0);
        Aterms aterms(beam);

        auto workunits = partition(tbl, gridconf);
        fmt::println("Nworkunits: {}", workunits.size());

        simple_benchmark(fmt::format("Invert {} px", i * 1000), 5, [&] {
            return invert<StokesI, float>(
                tbl, workunits, gridconf, aterms
            );
        });

        HostArray<StokesI<float>, 2> skymap(gridconf.grid().shape());

        simple_benchmark(fmt::format("Predict {} px", i * 1000), 10, [&] {
            predict<StokesI, float>(
                tbl, workunits, skymap, gridconf, aterms, DegridOp::Add
            );
            return 0;
        });
    }
}

TEST_CASE("Kernel size", "[kernelsize]") {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    DataTable tbl(TESTDATA, {});

    for (int kernelsize : {32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512}) {
        GridConfig gridconf {
            .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
            .paddingfactor=1, .kernelsize = kernelsize, .kernelpadding = 12
        };

        auto beam = Beam::Uniform<double>().gridResponse(gridconf.subgrid(), {0, 0}, 0);
        Aterms aterms(beam);

        auto workunits = partition(tbl, gridconf);
        fmt::println("Nworkunits: {}", workunits.size());

        simple_benchmark(fmt::format("Invert kernelsize: {} nvis: {}", kernelsize, tbl.size()), 10, [&] {
            return invert<StokesI, float>(
                tbl, workunits, gridconf, aterms
            );
        });

        HostArray<StokesI<float>, 2> skymap(gridconf.grid().shape());

        simple_benchmark(fmt::format("Predict kernelsize: {} nvis: {}", kernelsize, tbl.size()), 10, [&] {
            predict<StokesI, float>(
                tbl, workunits, skymap, gridconf, aterms, DegridOp::Add
            );
            return 0;
        });
    }
}