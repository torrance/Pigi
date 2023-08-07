#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <fmt/format.h>
#include <matplot/matplot.h>

#include "dft.h"
#include "invert.h"
#include "memory.h"
#include "mset.h"
#include "taper.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"

template <typename T>
std::vector<std::vector<T>> arrayToVectors(const HostSpan<StokesI<T>, 2> img, size_t padding) {
    std::vector<std::vector<T>> vecs(img.size(0) - 2 * padding);

    for (size_t ny = padding; ny < img.size(0) - padding; ++ny) {
        for (size_t nx = padding; nx < img.size(1) - padding; ++nx) {
            vecs[ny - padding].push_back(
                img[ny * img.size(0) + nx].I.real()
            );
        }
    }

    return vecs;
}

std::vector<UVDatum<double>> mkuvdata() {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> rand(0, 1);

    // Create a list of Ra/Dec sources
    std::vector<std::tuple<double, double>> sources;
    for (size_t i {}; i < 500; ++i) {
        double ra { deg2rad((rand(gen) - 0.5) * 30) };
        double dec { deg2rad((rand(gen) - 0.5) * 30) };
        sources.emplace_back(ra, dec);
    }

    std::vector<UVDatum<double>> uvdata;
    for (size_t i {}; i < 20000; ++i) {
        double u = rand(gen), v = rand(gen), w = rand(gen);

        // Scale uv to be in -500 <= +500 and w 0 < 500
        u = (u - 0.5) * 1000;
        v = (v - 0.5) * 1000;
        w*= 500;

        ComplexLinearData<double> data;
        for (auto [ra, dec] : sources) {
            double l { std::sin(ra) }, m = { std::sin(dec) };
            auto phase = cispi(-2 * (
                u * l + v * m + w * ndash(l, m)
            ));
            data += ComplexLinearData<double> {phase, 0, 0, phase};
        }

        uvdata.emplace_back(
            i, 0, u, v, w, LinearData<double>{1, 1, 1, 1}, data
        );
    }

    return uvdata;
}


TEST_CASE( "Arrays, Spans and H<->D transfers" ) {
    std::vector<int> v(8192, 1);

    HostSpan<int, 1> hs(v);
    REQUIRE( hs[0] == 1 );
    REQUIRE( hs[8191] == 1 );

    HostArray<int, 1> ha({8192});
    REQUIRE( ha[0] == 0 );
    REQUIRE( ha[8191] == 0 );

    ha = hs;
    REQUIRE( ha[0] == 1 );
    REQUIRE( ha[8191] == 1 );

    DeviceArray<int, 1> da(ha);

    ha.zero();
    REQUIRE( ha[0] == 0 );
    REQUIRE( ha[8191] == 0 );

    ha = da;
    REQUIRE( ha[0] == 1 );
    REQUIRE( ha[8191] == 1 );
}

TEMPLATE_TEST_CASE( "Coroutines", "", float, double) {
    // Config
    auto gridspec = GridSpec::fromScaleLM(1500, 1500, deg2rad(15. / 3600));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);
    int padding = 18;
    int wstep = 25;

    // Create dummy Aterms
    HostArray<ComplexLinearData<TestType>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    // Create tapers
    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);

    // Create uvdata
    auto _uvdata = mkuvdata();

    // Weight naturally
    for (auto& uvdatum : _uvdata) {
        uvdatum.weights = {1, 1, 1, 1};
        uvdatum.weights /= _uvdata.size();
    }

    // Cast to float or double
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : _uvdata) { uvdata.push_back((UVDatum<TestType>) uvdatum); }

    // Calculate expected
    HostArray<StokesI<TestType>, 2> expected({gridspec.Nx, gridspec.Ny});
    idft<StokesI<TestType>, TestType>(expected, uvdata, gridspec, 1);


    auto workunits = partition(
        uvdata, gridspec, subgridspec, padding, wstep, Aterms.asSpan()
    );

    auto img = invert<StokesI, TestType, HostArray<UVDatum<TestType>, 1>>(
        workunits, gridspec, taper, subtaper
    );

    double maxdiff {};
    for (size_t nx = 250; nx < 1250; ++nx) {
        for (size_t ny = 250; ny < 1250; ++ny) {
            auto idx = gridspec.gridToLinear(nx, ny);
            double diff = std::abs(expected[idx].I - img[idx].I);
            maxdiff = std::max(maxdiff, diff);
        }
    }
    fmt::println("Max diff: {:g}", maxdiff);
    REQUIRE( maxdiff < (std::is_same<float, TestType>::value ? 5e-5 : 2e-10) );
}