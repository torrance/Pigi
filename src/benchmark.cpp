#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>

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
    // Perform one warm up and keep the result to return
    auto ret = f();

    std::vector<double> timings;
    for (int n {}; n < N; ++n) {
        auto start = std::chrono::steady_clock::now();
        f();
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
    auto variance = std::transform_reduce(
        timings.begin(), timings.end(), 0., std::plus{}, [=] (auto timing) {
            return std::pow(timing - mean, 2);
        }
    );

    fmt::println(
        "Benchmark: {} ({} samples) mean: {:.6f} +/- {:.6f} s median: {:.6f} s",
        name, N, mean / 1000000, std::sqrt(variance) / 1000000, median / 1000000
    );

    return ret;
}

TEST_CASE("MSet reading and paritioning", "[io]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    HostArray<ComplexLinearData<double>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        TESTDATA,
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
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);

    HostArray<ComplexLinearData<TestType>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        TESTDATA,
        {.chanlow = 0, .chanhigh = 383}
    );

    // Convert to TestType precision
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : mset) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    auto workunits = partition(uvdata, gridspec, subgridspec, 18, 25, Aterms);

    simple_benchmark("Invert", 1, [&] {
        return invert<StokesI, TestType>(
            workunits, gridspec, taper, subtaper
        );
    });
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    auto gridspec = GridSpec::fromScaleLM(8000, 8000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);

    HostArray<ComplexLinearData<TestType>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        TESTDATA,
        {.chanlow = 0, .chanhigh = 383}
    );

    // Convert to TestType precision
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : mset) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    auto workunits = partition(uvdata, gridspec, subgridspec, 18, 25, Aterms);

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap({gridspec.Nx, gridspec.Ny});

    simple_benchmark("Predict", 1, [&] {
        predict<StokesI<TestType>, TestType>(
            workunits, skymap, gridspec, taper, subtaper, DegridOp::Replace
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

    auto uvdata_d = DeviceArray<UVDatum<TestType>, 1>::fromVector(uvdata_h);

    auto subgridspec = GridSpec::fromScaleLM(96, 96, deg2rad(15. / 3600));
    UVWOrigin<TestType> origin {0, 0, 0};

    HostArray<ComplexLinearData<TestType>, 2> Aterm_h({subgridspec.Nx, subgridspec.Ny});
    Aterm_h.fill({1, 0, 0, 1});
    DeviceArray<ComplexLinearData<TestType>, 2> Aterm_d(Aterm_h);

    DeviceArray<StokesI<TestType>, 2> subgrid({subgridspec.Nx, subgridspec.Ny});

    simple_benchmark("gpudift", 10, [&] {
        for (size_t i {}; i < 25; ++i) {
            gpudift<StokesI<TestType>, TestType>(
                subgrid, Aterm_d, Aterm_d, origin, uvdata_d, subgridspec, false
            );
        }
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return true;
    });
}

TEMPLATE_TEST_CASE("gpudft kernel", "[gpudft]", float, double) {
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

    auto uvdata_d = DeviceArray<UVDatum<TestType>, 1>::fromVector(uvdata_h);

    auto subgridspec = GridSpec::fromScaleLM(96, 96, deg2rad(15. / 3600));
    UVWOrigin<TestType> origin {0, 0, 0};

    DeviceArray<ComplexLinearData<TestType>, 2> subgrid({subgridspec.Nx, subgridspec.Ny});

    simple_benchmark("gpudft", 10, [&] {
        for (size_t i {}; i < 25; ++i) {
            gpudft<TestType>(
                uvdata_d, origin, subgrid, subgridspec, DegridOp::Replace
            );
        }
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
        return true;
    });
}