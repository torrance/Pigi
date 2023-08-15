#include <cmath>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>

#include "clean.h"
#include "dft.h"
#include "degridder.h"
#include "fits.h"
#include "invert.h"
#include "memory.h"
#include "mset.h"
#include "predict.h"
#include "taper.h"
#include "util.h"
#include "uvdatum.h"
#include "workunit.h"


TEST_CASE( "Arrays, Spans and H<->D transfers", "[memory]" ) {
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

TEST_CASE("Measurement Set & Partition", "[mset]") {
    auto gridspec = GridSpec::fromScaleLM(1000, 1000, std::sin(deg2rad(15. / 60)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    HostArray<ComplexLinearData<double>, 2> Aterms({96, 96});
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        "/home/torrance/testdata/1215555160/1215555160.ms",
        {.chanlow = 0, .chanhigh = 11}
    );

    auto workunits = partition(
        *mset.uvdata(), gridspec, subgridspec, 18, 25, Aterms
    );

    size_t n {};
    for (auto& workunit : workunits) {
        n += workunit.data.size();
    }

    REQUIRE( n == 2790000 );
}

TEMPLATE_TEST_CASE( "Invert", "[invert]", float, double) {
    // Config
    auto gridspec = GridSpec::fromScaleLM(1500, 1500, std::sin(deg2rad(15. / 3600)));
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
    std::vector<UVDatum<double>> uvdata64;
    {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<double> rand(0, 1);

        // Create a list of Ra/Dec sources
        std::vector<std::tuple<double, double>> sources;
        for (size_t i {}; i < 250; ++i) {
            double ra { deg2rad((rand(gen) - 0.5) * 5) };
            double dec { deg2rad((rand(gen) - 0.5) * 5) };
            sources.emplace_back(ra, dec);
        }

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

            // TODO: use emplace_back() when we can upgrade Clang
            uvdata64.push_back(
                {i, 0, u, v, w, LinearData<double>{1, 1, 1, 1}, data}
            );
        }
    }

    // Weight naturally
    for (auto& uvdatum : uvdata64) {
        uvdatum.weights = {1, 1, 1, 1};
        uvdatum.weights /= uvdata64.size();
    }

    // Calculate expected at double precision
    HostArray<StokesI<double>, 2> expected({gridspec.Nx, gridspec.Ny});
    idft<StokesI<double>, double>(expected, uvdata64, gridspec, 1);

    // Cast to float or double
    std::vector<UVDatum<TestType>> uvdata;
    for (const auto& uvdatum : uvdata64) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    auto workunits = partition(
        uvdata, gridspec, subgridspec, padding, wstep, Aterms
    );

    auto img = invert<StokesI, TestType>(
        workunits, gridspec, taper, subtaper
    );

    double maxdiff {};
    for (size_t nx = 250; nx < 1250; ++nx) {
        for (size_t ny = 250; ny < 1250; ++ny) {
            auto idx = gridspec.gridToLinear(nx, ny);
            double diff = std::abs(
                expected[idx].I - std::complex<double>(img[idx].I)
            );
            maxdiff = std::max(maxdiff, diff);
        }
    }
    fmt::println("Max diff: {:g}", maxdiff);
    REQUIRE( maxdiff < (std::is_same<float, TestType>::value ? 1e-5 : 2e-10) );
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    auto gridspec = GridSpec::fromScaleUV(2000, 2000, 1);

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap({gridspec.Nx, gridspec.Ny});

    std::mt19937 gen(1234);
    std::uniform_int_distribution<int> randints(700, 1300);

    for (size_t i {}; i < 1000; ++i) {
        int x {randints(gen)}, y {randints(gen)};
        skymap[gridspec.gridToLinear(x, y)] = StokesI<TestType> {TestType(1)};
    }

    std::uniform_real_distribution<TestType> randfloats(0, 1);

    // Create empty UVDatum
    std::vector<UVDatum<TestType>> uvdata;
    for (size_t i {}; i < 5000; ++i) {
        TestType u {randfloats(gen)}, v {randfloats(gen)}, w {randfloats(gen)};
        u = (u - 0.5) * 1000;
        v = (v - 0.5) * 1000;
        w = (w - 0.5) * 500;

        // TODO: use emplace_back() when we can upgrade Clang
        uvdata.push_back({
            i, 0, u, v, w,
            LinearData<TestType> {1, 1, 1, 1},
            ComplexLinearData<TestType> {0, 0, 0, 0}
        });
    }

    // Calculate expected at double precision
    std::vector<UVDatum<double>> expected;
    {
        // Find non-empty pixels
        std::vector<size_t> idxs;
        for (size_t i {}; i < skymap.size(); ++i) {
            if (std::abs(skymap[i].I) != 0) idxs.push_back(i);
        }

        // For each UVDatum, sum over non-empty pixels
        for (const auto& uvdatum : uvdata) {
            UVDatum<double> uvdatum64 = static_cast<UVDatum<double>>(uvdatum);
            for (auto idx : idxs) {
                StokesI<double> cell {skymap[idx].I};
                auto [l, m] = gridspec.linearToSky<double>(idx);
                cell *= cispi(
                    -2 * (uvdatum64.u * l + uvdatum64.v * m + uvdatum64.w * ndash(l, m))
                );
                uvdatum64.data += (ComplexLinearData<double>) cell;
            }
            expected.push_back(uvdatum64);
        }
    }

    // Predict using IDG
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);
    auto taper = kaiserbessel<TestType>(gridspec);
    auto subtaper = kaiserbessel<TestType>(subgridspec);
    int padding {17};
    int wstep {25};
    HostArray<ComplexLinearData<TestType>, 2> Aterms({subgridspec.Nx, subgridspec.Ny});
    Aterms.fill({1, 0, 0, 1});

    auto workunits = partition(
        uvdata, gridspec, subgridspec, padding, wstep, Aterms
    );

    predict<StokesI<TestType>, TestType>(
        workunits, skymap, gridspec, taper, subtaper, DegridOp::Replace
    );

    // Flatten workunits back into uvdata and sort back to original order
    for (const auto& workunit : workunits) {
        for (const auto& uvdatum : workunit.data) {
            uvdata[uvdatum.row] = uvdatum;
        }
    }

    double maxdiff {};
    for (size_t i {}; i < uvdata.size(); ++i) {
        auto diff = uvdata[i].data;
        diff -= expected[i].data;

        maxdiff = std::max<double>(
            maxdiff,
            std::abs(diff.xx) + std::abs(diff.yx) +
            std::abs(diff.xy) + std::abs(diff.yy)
        );
    }

    fmt::println("Prediction max diff: {}", maxdiff);
    REQUIRE( maxdiff < (std::is_same<float, TestType>::value ? 1e-3 : 2e-9) );
}

TEST_CASE("Clean", "[clean]") {
    auto gridspec = GridSpec::fromScaleLM(1000, 1000, deg2rad(1. / 60));
    HostArray<StokesI<double>, 2> expected({1000, 1000});

    std::mt19937 gen(1234);
    std::uniform_int_distribution<size_t> randidx(0, expected.size());
    std::uniform_real_distribution<double> randflux(0, 1);

    for (size_t i {}; i < 25; ++i) {
        expected[randidx(gen)] = StokesI<double> {randflux(gen)};
    }

    clean::PSF expectedPSF {deg2rad(5. / 60), deg2rad(2. / 60), deg2rad(34.5)};

    auto dirtyPSF = expectedPSF.template draw<StokesI<double>>(gridspec);

    // TODO: These casts are mess. Find a consistent type.
    HostArray<double, 2> dirtyPSF_real(dirtyPSF.shape());
    HostArray<std::complex<double>, 2> dirtyPSF_complex(dirtyPSF.shape());
    for (size_t i {}; i < dirtyPSF.size(); ++i) {
        dirtyPSF_real[i] = dirtyPSF[i].real();
        dirtyPSF_complex[i] = dirtyPSF[i].I;
    }

    clean::convolve(expected, dirtyPSF_complex);

    HostArray<StokesI<double>, 2> img {expected};
    auto [components, iter] = clean::major(
        img, gridspec,
        dirtyPSF, gridspec,
        {.mgain = 0.991}
    );

    auto fittedPSF = clean::fitpsf(dirtyPSF_real, gridspec)
        .template draw<std::complex<double>>(gridspec);

    clean::convolve(components, fittedPSF);

    double maxdiff {};

    for (size_t i{}; i < expected.size(); ++i) {
        maxdiff = std::max<double>(
            std::abs(expected[i].I - components[i].I), maxdiff
        );
    }
    fmt::println("Max diff: {}", maxdiff);
    REQUIRE(maxdiff < 0.01);
}