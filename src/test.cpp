#include <cmath>
#include <cstdlib>
#include <generator>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "beam.h"
#include "clean.h"
#include "dft.h"
#include "degridder.h"
#include "fits.h"
#include "invert.h"
#include "memory.h"
#include "mset.h"
#include "psf.h"
#include "predict.h"
#include "taper.h"
#include "util.h"
#include "uvdatum.h"
#include "weighter.h"
#include "workunit.h"

const char* TESTDATA = getenv("TESTDATA");

TEST_CASE( "Arrays, Spans and H<->D transfers", "[memory]" ) {
    std::vector<int> v(8192, 1);

    HostSpan<int, 1> hs(v);
    REQUIRE( hs[0] == 1 );
    REQUIRE( hs[8191] == 1 );

    HostArray<int, 1> ha {8192};
    REQUIRE( ha[0] == 0 );
    REQUIRE( ha[8191] == 0 );

    ha = hs;
    REQUIRE( ha[0] == 1 );
    REQUIRE( ha[8191] == 1 );

    DeviceArray<int, 1> da {ha};

    ha.zero();
    REQUIRE( ha[0] == 0 );
    REQUIRE( ha[8191] == 0 );

    ha = da;
    REQUIRE( ha[0] == 1 );
    REQUIRE( ha[8191] == 1 );
}

TEST_CASE("Matrix tests", "[matrix]") {
    ComplexLinearData<double> A {12, 44, 3, 32};
    ComplexLinearData<double> B {{42, 1}, {4, 7}, {31, -3}, {-3, -5}};

    auto C = matmul(B, A);
    REQUIRE(
        C == ComplexLinearData<double>{{1868, -120}, {-84, -136}, {1118, -93}, {-84, -139}}
    );

    C = matmul(A, B);
    REQUIRE(
        C == ComplexLinearData<double>{{516, 33}, {1976, 268}, {363, -51}, {1268, -292}}
    );

    C = A.inv().adjoint();
    REQUIRE((
        thrust::abs(C.xx - 0.126984) < 1e-5 &&
        thrust::abs(C.yx - -0.0119048) < 1e-5 &&
        thrust::abs(C.xy - -0.174603) < 1e-5 &&
        thrust::abs(C.yy - 0.047619) < 1e-5
    ));

    C = B.adjoint().inv();
    REQUIRE((
        thrust::abs(C.xx - thrust::complex<double>{0.0117647, -0.000309598}) < 1e-5 &&
        thrust::abs(C.yx - thrust::complex<double>{0.028483, 0.0560372}) < 1e-5 &&
        thrust::abs(C.xy - thrust::complex<double>{0.0162539, -0.000773994}) < 1e-5 &&
        thrust::abs(C.yy - thrust::complex<double>{-0.0472136, -0.0704334}) < 1e-5
    ));
}

TEST_CASE("Measurement Set & Partition", "[mset]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    auto gridspec = GridSpec::fromScaleLM(1000, 1000, std::sin(deg2rad(15. / 3600)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);

    HostArray<ComplexLinearData<double>, 2> Aterms {96, 96};
    Aterms.fill({1, 0, 0, 1});

    MeasurementSet mset(
        TESTDATA, 0, 11, 0, std::numeric_limits<double>::max()
    );

    auto workunits = partition(
        mset, gridspec, subgridspec, 18, 25, Aterms
    );

    size_t n {};
    for (auto& workunit : workunits) {
        n += workunit.data.size();
    }

    REQUIRE( n == 2790000 );
}

// Catch2 doesn't seem to support namespace separators
// so we rename these here
template <typename Q> using UniformBeam = Beam::Uniform<Q>;
template <typename Q> using GaussianBeam = Beam::Gaussian<Q>;
template <typename Q> using MWABeam = Beam::MWA<Q>;

TEMPLATE_TEST_CASE_SIG(
    "Invert", "[invert]",
    ((typename Q, typename BEAM, int THRESHOLDF, int THRESHOLDP), Q, BEAM, THRESHOLDF, THRESHOLDP),
    (float, (UniformBeam<float>), 2, -5),
    (double, (UniformBeam<double>), 2, -10),
    (float, (GaussianBeam<float>), 2, -5),
    (double, (GaussianBeam<double>), 2, -10),
    (float, (MWABeam<float>), 7, -4),
    (double, (MWABeam<double>), 8, -6)
) {
    // Config
    auto gridspec = GridSpec::fromScaleLM(1500, 1500, std::sin(deg2rad(2.2 / 60)));
    auto subgridspec = GridSpec::fromScaleUV(96, 96, gridspec.scaleuv);
    int padding = 18;
    int wstep = 25;
    double freq = 150e6;
    AzEl gridorigin {0, pi_v<double> / 2};

    // Create Aterms
    BEAM beam;
    if constexpr(std::is_same<Beam::Gaussian<Q>, BEAM>::value) {
        beam = BEAM(gridorigin, deg2rad(25.));
    }
    auto Aterm = beam.gridResponse(subgridspec, gridorigin, freq);

    // Create tapers
    auto taper = kaiserbessel<Q>(gridspec);
    auto subtaper = kaiserbessel<Q>(subgridspec);

    // Create uvdata
    std::vector<UVDatum<double>> uvdata64;
    {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<double> rand(0, 1);

        // Create a list of Ra/Dec sources
        std::vector<std::tuple<double, double, ComplexLinearData<double>>> sources;
        for (size_t i {}; i < 250; ++i) {
            double l { std::sin( deg2rad((rand(gen) - 0.5) * 50) ) };
            double m { std::sin( deg2rad((rand(gen) - 0.5) * 50) ) };

            auto jones = static_cast<ComplexLinearData<double>>(
                beam.pointResponse(lmToAzEl(l, m, gridorigin), freq)
            );
            sources.emplace_back(l, m, jones);
        }

        for (size_t i {}; i < 20000; ++i) {
            double u = rand(gen), v = rand(gen), w = rand(gen);

            // Scale uv to be in -500 <= +500 and w 0 < 500
            u = (u - 0.5) * 200;
            v = (v - 0.5) * 200;
            w*= 200;

            ComplexLinearData<double> data;
            for (auto [l, m, jones] : sources) {
                auto phase = cispi(-2 * (
                    u * l + v * m + w * ndash(l, m)
                ));

                ComplexLinearData<double> cell {phase, 0, 0, phase};
                data += matmul(matmul(jones, cell), jones.adjoint());
            }

            // TODO: use emplace_back() when we can upgrade Clang
            uvdata64.push_back(
                {i, 0, u, v, w, LinearData<double>{1, 1, 1, 1}, data}
            );
        }
    }

    // Weight naturally
    const Natural<double> weighter(uvdata64, gridspec);
    applyWeights(weighter, uvdata64);

    // Calculate expected at double precision
    HostArray<StokesI<double>, 2> expected {gridspec.Nx, gridspec.Ny};
    {
        auto jones = static_cast<HostArray<ComplexLinearData<double>, 2>>(
            beam.gridResponse(gridspec, gridorigin, freq)
        );
        idft<StokesI<double>, double>(expected, jones, uvdata64, gridspec);
    }

    // Cast to float or double
    auto uvdata = [&] () -> std::generator<UVDatum<Q>> {
        for (const auto& uvdatum : uvdata64) {
            co_yield static_cast<UVDatum<Q>>(uvdatum);
        }
    }();

    auto workunits = partition(
        uvdata, gridspec, subgridspec, padding, wstep, Aterm
    );

    auto img = invert<StokesI, Q>(
        workunits, gridspec, taper, subtaper
    );

    // Correct for beam
    auto jonesgrid = beam.gridResponse(gridspec, gridorigin, freq);
    HostArray<StokesI<Q>, 2> power {gridspec.Nx, gridspec.Ny};
    for (size_t i {}; i < gridspec.size(); ++i) {
        img[i] /= StokesI<Q>::beamPower(jonesgrid[i], jonesgrid[i]);
    }

    double maxdiff {-1};
    for (size_t nx = 250; nx < 1250; ++nx) {
        for (size_t ny = 250; ny < 1250; ++ny) {
            auto idx = gridspec.gridToLinear(nx, ny);
            double diff = thrust::abs(
                expected[idx].I - thrust::complex<double>(img[idx].I)
            );
            maxdiff = std::max(maxdiff, diff);
        }
    }
    fmt::println("Max diff: {:g}", maxdiff);
    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < THRESHOLDF * std::pow(10, THRESHOLDP));
}

TEMPLATE_TEST_CASE("Predict", "[predict]", float, double) {
    auto gridspec = GridSpec::fromScaleUV(2000, 2000, 1);

    // Create skymap
    HostArray<StokesI<TestType>, 2> skymap {gridspec.Nx, gridspec.Ny};

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
        u = (u - 0.5) * 250;
        v = (v - 0.5) * 250;
        w = (w - 0.5) * 100;

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
            if (thrust::abs(skymap[i].I) != 0) idxs.push_back(i);
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
    HostArray<ComplexLinearData<TestType>, 2> Aterms {subgridspec.Nx, subgridspec.Ny};
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

    double maxdiff {-1};
    for (size_t i {}; i < uvdata.size(); ++i) {
        auto diff = uvdata[i].data;
        diff -= expected[i].data;

        maxdiff = std::max<double>(
            maxdiff,
            thrust::abs(diff.xx) + thrust::abs(diff.yx) +
            thrust::abs(diff.xy) + thrust::abs(diff.yy)
        );
    }

    fmt::println("Prediction max diff: {}", maxdiff);
    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < (std::is_same<float, TestType>::value ? 1e-3 : 3e-9) );
}

TEST_CASE("Clean", "[clean]") {
    auto gridspec = GridSpec::fromScaleLM(1000, 1000, deg2rad(1. / 60));

    // Use 4 channelgroups
    const int N = 4;

    // Create 4 model arrays...
    std::vector<HostArray<StokesI<double>, 2>> models;
    for (int n {}; n < N; ++n) {
        models.push_back(
           HostArray<StokesI<double>, 2> {1000, 1000}
        );
    }

    // and then populate with 25 random point sources with linear flux distributions
    std::mt19937 gen(1234);
    std::uniform_int_distribution<size_t> randidx(0, gridspec.size());
    std::uniform_real_distribution<double> randflux(0, 1);
    std::uniform_real_distribution<double> randgradient(-0.2, 0.2);

    for (size_t i {}; i < 25; ++i) {
        auto idx = randidx(gen);
        auto c = randflux(gen);
        auto m = randgradient(gen);

        for (int n {}; n < N; ++n) {
            models[n][idx] = c + m * n;
        }
    }

    // Create PSF and and psf map
    PSF<double> expectedPSF {deg2rad(5. / 60), deg2rad(2. / 60), deg2rad(34.5)};
    auto dirtyPSF = expectedPSF.draw(gridspec);

    HostArray<StokesI<double>, 2> expectedSum {gridspec.Nx, gridspec.Ny};

    // Convole the modles, and combine these into ChannelGroup vector
    std::vector<ChannelGroup<StokesI, double>> channelgroups;
    for (int n {}; n < N; ++n) {
        auto expected = convolve(models[n], dirtyPSF);
        expectedSum += expected;

        channelgroups.push_back(ChannelGroup<StokesI, double>{
            .channelIndex = n,
            .midfreq = 1e6 * (n + 1),
            .psf = HostArray<thrust::complex<double>, 2> {dirtyPSF},
            .residual = std::move(expected)
        });
    }
    expectedSum /= StokesI<double>(4);

    // Find peak max, which we use as the final test value
    double maxInit {};
    for (size_t i {}; i < expectedSum.size(); ++i) {
        maxInit = std::max(maxInit, expectedSum[i].I.real());
    };

    auto [components, iter, _] = clean::major(
        channelgroups, gridspec, gridspec,
        0.1, 0.991, 0, 0, std::numeric_limits<size_t>::max()
    );

    auto fittedPSF = PSF<double>(dirtyPSF, gridspec).draw(gridspec);

    // Combine the convolved, component images to produce a restored image
    HostArray<StokesI<double>, 2> restoredSum {gridspec.Nx, gridspec.Ny};
    for (auto& component : components) {
        HostArray<StokesI<double>, 2> componentMap {gridspec.Nx, gridspec.Ny};
        for (auto& [idx, val] : component) {
            componentMap[idx] += val;
        }
        restoredSum += convolve(componentMap, fittedPSF);
    }
    restoredSum /= StokesI<double>(4);

    // Compute the test result
    double maxdiff {};
    for (size_t i{}; i < expectedSum.size(); ++i) {
        maxdiff = std::max<double>(
            thrust::abs(expectedSum[i].I - restoredSum[i].I), maxdiff
        );
    }
    fmt::println("Max diff: {}", maxdiff);
    REQUIRE(maxdiff < maxInit * 0.01);
}