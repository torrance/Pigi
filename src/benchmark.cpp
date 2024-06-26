#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>

#include "aterms.h"
#include "clean.h"
#include "degridder.h"
#include "gridspec.h"
#include "gridder.h"
#include "invert.h"
#include "mset.h"
#include "memory.h"
#include "outputtypes.h"
#include "predict.h"
#include "taper.h"
#include "uvdatum.h"
#include "workunit.h"

const char* TESTDATA = getenv("TESTDATA");

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
    auto mean = std::accumulate(timings.begin(), timings.end(), 0.) / N;
    double variance {};
    for (auto timing : timings) {
        variance += std::pow(timing - mean, 2) / N;
    }

    fmt::println(
        "Benchmark: {} ({} samples) mean: {:.6f} +/- {:.6f} s median: {:.6f} s",
        name, N, mean / 1000000, std::sqrt(variance) / 1000000, median / 1000000
    );

    return ret;
}

TEST_CASE("MSet reading and paritioning", "[io]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .kernelsize = 96, .kernelpadding = 18
    };

    HostArray<ComplexLinearData<double>, 2> Aterm {gridconf.subgrid().shape()};
    Aterm.fill({1, 0, 0, 1});

    MeasurementSet mset(
        {TESTDATA}, MeasurementSet::DataColumn::data,
        0, 191, 0, std::numeric_limits<double>::max()
    );

    auto uvdata = simple_benchmark("MSet read", 1, [&] {
        return mset.data<double>();
    });

    auto aterms = mkAterms<double>(mset, gridconf.subgrid(), 30, mset.phaseCenter());

    auto workunits = simple_benchmark("Partition", 3, [&] {
        return partition<double>(uvdata, gridconf, aterms);
    });
}

TEMPLATE_TEST_CASE("Invert", "[invert]", float, double) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .kernelsize = 96, .kernelpadding = 18
    };

    HostArray<ComplexLinearData<TestType>, 2> Aterm {gridconf.subgrid().shape()};
    Aterm.fill({1, 0, 0, 1});

    MeasurementSet mset(
        {TESTDATA}, MeasurementSet::DataColumn::data,
        0, 383, 0, std::numeric_limits<double>::max()
    );

    // Convert to TestType precision
    auto uvdata = mset.data<TestType>();
    auto workunits = partition(uvdata, gridconf, Aterms<TestType>(Aterm));

    simple_benchmark("Invert", 5, [&] {
        return invert<StokesI, TestType>(
            workunits, gridconf
        );
    });
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx = 8000, .imgNy = 8000, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .kernelsize = 96, .kernelpadding = 18
    };

    HostArray<ComplexLinearData<TestType>, 2> Aterm {gridconf.subgrid().shape()};
    Aterm.fill({1, 0, 0, 1});

    MeasurementSet mset(
        {TESTDATA}, MeasurementSet::DataColumn::data,
        0, 383, 0, std::numeric_limits<double>::max()
    );

    auto uvdata = mset.data<TestType>();
    auto workunits = partition(uvdata, gridconf, Aterms<TestType>(Aterm));

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap {gridconf.grid().shape()};

    simple_benchmark("Predict", 5, [&] {
        predict<StokesI<TestType>, TestType>(
            workunits, skymap, gridconf, DegridOp::Add
        );
        return true;
    });
}

TEMPLATE_TEST_CASE("gpudift kernel", "[gpudift]", float, double) {
    std::vector<UVDatum<TestType>> uvdata_h;

    std::mt19937 gen(1234);
    std::uniform_real_distribution<TestType> rand;

    for (size_t i {}; i < 25000; ++i) {
        TestType u { (rand(gen) - TestType(0.5)) * 100 };
        TestType v { (rand(gen) - TestType(0.5)) * 100 };
        TestType w { (rand(gen) - TestType(0.5)) * 100 };

        uvdata_h.push_back(UVDatum<TestType> {
            0, 0, u, v, w,
            {rand(gen), rand(gen), rand(gen), rand(gen)},
            {
                {rand(gen), rand(gen)}, {rand(gen), rand(gen)},
                {rand(gen), rand(gen)}, {rand(gen), rand(gen)}
            }
        });
    }

    std::vector<DeviceArray<UVDatum<TestType>, 1>> uvdata_ds;
    for (size_t i {}; i < 25; ++i) {
        uvdata_ds.push_back(
            DeviceArray<UVDatum<TestType>, 1> {uvdata_h}
        );
    }

    auto subgridspec = GridSpec::fromScaleLM(96, 96, deg2rad(15. / 3600));
    UVWOrigin<TestType> origin {0, 0, 0};

    HostArray<ComplexLinearData<TestType>, 2> Aterm_h {subgridspec.Nx, subgridspec.Ny};
    Aterm_h.fill({1, 0, 0, 1});
    DeviceArray<ComplexLinearData<TestType>, 2> Aterm_d {Aterm_h};

    DeviceArray<StokesI<TestType>, 2> subgrid {subgridspec.Nx, subgridspec.Ny};

    simple_benchmark("gpudift", 1, [&] {
        for (size_t i {}; i < 25; ++i) {
            gpudift<StokesI<TestType>, TestType>(
                subgrid, Aterm_d, Aterm_d, origin, uvdata_ds[i], subgridspec, false
            );
        }
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return true;
    });
}

TEMPLATE_TEST_CASE("gpudft kernel", "[gpudft]", float, double) {
    std::vector<UVDatum<TestType>, ManagedAllocator<UVDatum<TestType>>> uvdata;

    std::mt19937 gen(1234);
    std::uniform_real_distribution<TestType> rand;

    for (size_t i {}; i < 25000; ++i) {
        TestType u { (rand(gen) - TestType(0.5)) * 100 };
        TestType v { (rand(gen) - TestType(0.5)) * 100 };
        TestType w { (rand(gen) - TestType(0.5)) * 100 };

        uvdata.push_back(UVDatum<TestType> {
            0, 0, u, v, w,
            {rand(gen), rand(gen), rand(gen), rand(gen)},
            {
                {rand(gen), rand(gen)}, {rand(gen), rand(gen)},
                {rand(gen), rand(gen)}, {rand(gen), rand(gen)}
            }
        });
    }

    // Assemble pointers for each uvdatum and send to device
    std::vector<UVDatum<TestType>*> uvdata_ptrs_h;
    for (auto& uvdatum : uvdata) uvdata_ptrs_h.push_back(&uvdatum);
    DeviceArray<UVDatum<TestType>*, 1> uvdata_ptrs(uvdata_ptrs_h);

    HIPCHECK( hipMemPrefetchAsync(uvdata.data(), uvdata.size(), 0, 0) );

    auto subgridspec = GridSpec::fromScaleLM(96, 96, deg2rad(15. / 3600));
    HostArray<ComplexLinearData<TestType>, 2> subgrid_h {subgridspec.Nx, subgridspec.Ny};
    for (size_t i {}; i < subgridspec.size(); ++i) {
        subgrid_h[i] = {
            {rand(gen), rand(gen)},
            {rand(gen), rand(gen)},
            {rand(gen), rand(gen)},
            {rand(gen), rand(gen)}
        };
    }

    std::vector<DeviceArray<ComplexLinearData<TestType>, 2>> subgrid_ds;
    for (size_t i {}; i < 25; ++i) {
        subgrid_ds.push_back(
            DeviceArray<ComplexLinearData<TestType>, 2> {subgrid_h}
        );
    }

    UVWOrigin<TestType> origin {0, 0, 0};
    DeviceArray<ComplexLinearData<TestType>, 1> output(25000);

    simple_benchmark("gpudft", 1, [&] {
        for (auto& subgrid : subgrid_ds) {
            gpudft<TestType>(
                output, uvdata_ptrs, origin, subgrid, subgridspec, DegridOp::Add
            );
        }
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return true;
    });
}