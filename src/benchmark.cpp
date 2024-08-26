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

auto TESTDATA = [] () -> const std::vector<casacore::MeasurementSet> {
    if (getenv("TESTDATA") == nullptr) return {};

    // TESTDATA may include multiple, space-separated paths
    std::string paths = getenv("TESTDATA");

    std::vector<casacore::MeasurementSet> msets;
    for (size_t i {}, len {}; i + len <= paths.size(); ++len) {
        if (i + len == paths.size() || paths[i + len] == ' ') {
            msets.push_back({paths.substr(i, len)});
            i = i + len + 1;
            len = 0;
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

TEMPLATE_TEST_CASE("(De)gridder kernels", "[kernels]", float) {
    if (TESTDATA.empty()) { SKIP("TESTDATA path not provided"); }

    DataTable tbl(TESTDATA, {});

    for (const int kpx : {32, 48, 64, 96, 128}) {
        GridConfig gridconf {
            .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
            .kernelsize = kpx, .kernelpadding = 12
        };

        auto workunits = partition(tbl, gridconf);

        // Create uvws
        HostArray<std::array<double, 3>, 1> uvws_h(workunits.size());
        for (size_t i {}; auto w : workunits) uvws_h[i++] = {w.u, w.v, w.w};

        // Create aterms
        auto beam_h = Beam::Uniform<double>().gridResponse(
            gridconf.subgrid(), {0, 0}, 0
        );
        DeviceArray<ComplexLinearData<double>, 2> beam_d(beam_h);
        HostArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> aterms_h(workunits.size());
        for (auto& aterm : aterms_h) aterm = beam_d;

        // Copy to device
        DeviceArray<TestType, 2> taper_d {pswf<TestType>(gridconf.subgrid())};
        DeviceArray<double, 1> lambdas_d(tbl.lambdas());
        DeviceArray<std::array<double, 3>, 1> uvws_d(uvws_h);
        DeviceArray<WorkUnit, 1> workunits_d(workunits);
        DeviceArray<ComplexLinearData<float>, 2> data_d(tbl.data());
        DeviceArray<LinearData<float>, 2> weights_d(tbl.weights());
        DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> alefts_d(aterms_h);
        DeviceArray<DeviceSpan<ComplexLinearData<double>, 2>, 1> arights_d(aterms_h);

        // Allocate subgrid
        DeviceArray<StokesI<TestType>, 3> subgrids_d(std::array<long long, 3>{
            gridconf.subgrid().Nx, gridconf.subgrid().Ny, (long long) workunits.size()
        });

        HIPCHECK( hipDeviceSynchronize() );

        simple_benchmark(fmt::format(
            "Gridder kernel={}px nworkunits={}", kpx, workunits.size()
        ), 1, [&] {
            gridder<StokesI<TestType>, TestType>(
                subgrids_d, workunits_d, uvws_d, data_d, weights_d, gridconf.subgrid(),
                lambdas_d, taper_d, alefts_d, arights_d, 0, false
            );
            return true;
        });

        simple_benchmark(fmt::format(
            "Degridder kernel={}px nworkunits={}", kpx, workunits.size()
        ), 1, [&] {
            degridder<StokesI<TestType>, TestType>(
                data_d, subgrids_d, workunits_d, uvws_d, lambdas_d, taper_d,
                alefts_d, arights_d, gridconf.subgrid(), 0, DegridOp::Add
            );
            return true;
        });
    }
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