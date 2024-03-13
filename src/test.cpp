#include <cmath>
#include <cstdlib>
#include <generator>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "aterms.h"
#include "beam.h"
#include "clean.h"
#include "dft.h"
#include "degridder.h"
#include "fits.h"
#include "invert.h"
#include "memory.h"
#include "mset.h"
#include "phaserotate.h"
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

    copy(ha, hs);
    REQUIRE( ha[0] == 1 );
    REQUIRE( ha[8191] == 1 );

    DeviceArray<int, 1> da {ha};

    ha.zero();
    REQUIRE( ha[0] == 0 );
    REQUIRE( ha[8191] == 0 );

    copy(ha, da);
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

TEST_CASE("Coordinates", "[coordinates]") {
    RaDec gridorigin {.ra=deg2rad(156.), .dec=deg2rad(-42.)};
    double mjd = 58290.621412037035;
    AzEl azelorigin = radecToAzel(gridorigin, mjd, Beam::MWA<double>::origin);

    {
        // Calculated using Astropy
        AzEl expected {deg2rad(229.97110571 - 360), deg2rad(14.81615802)};

        REQUIRE(std::abs(azelorigin.az - expected.az) < 1e-5);
        REQUIRE(std::abs(azelorigin.el - expected.el) < 1e-5);
    }

    double scalelm = std::sin(deg2rad(15. / 3600.));
    int lpx {3500}, mpx {5500};
    int N {8000};
    gridorigin = {.ra=deg2rad(17.4208333333333), .dec=deg2rad(-45.7819444444444)};

    auto radec = lmToRaDec((lpx - N / 2) * scalelm, (mpx - N / 2) * scalelm, gridorigin);
    {
        // Calculcated using Astropy
        RaDec expected {.ra=deg2rad(20.12114593), .dec=deg2rad(-39.48407916)};

        REQUIRE(std::abs(radec.ra - expected.ra) < 1e-5);
        REQUIRE(std::abs(radec.dec - expected.dec) < 1e-5);
    }

    auto [l, m] = RaDecTolm(radec, gridorigin);
    REQUIRE(std::abs(l - (lpx - N / 2) * scalelm) < 1e-9);
    REQUIRE(std::abs(m - (mpx - N / 2) * scalelm) < 1e-9);
}

TEST_CASE("Utility functions", "[utility]") {
    auto smallGridspec = GridSpec::fromScaleUV(128, 128, 1);
    auto largeGridspec = GridSpec::fromScaleUV(1000, 1000, 1);

    PSF<double> psf(deg2rad(25.), deg2rad(20.), deg2rad(31.));
    auto smallGaussian = psf.draw(smallGridspec);
    auto largeGaussian = rescale(smallGaussian, smallGridspec, largeGridspec);

    auto expectedGaussian = psf.draw(largeGridspec);

    double maxdiff {-1};
    for (size_t i {}; i < largeGridspec.size(); ++i) {
        maxdiff = std::max(maxdiff, thrust::abs(largeGaussian[i] - expectedGaussian[i]));
    }

    fmt::println("Maxdiff = {}", maxdiff);

    REQUIRE(maxdiff != -1);
    REQUIRE(maxdiff < 2e-2);
}

TEMPLATE_TEST_CASE("Phase rotation", "[phaserotation]", float, double) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    MeasurementSet mset(
        {TESTDATA}, 0, 11, 0, std::numeric_limits<double>::max()
    );

    std::vector<UVDatum<TestType>> uvdata;
    for (auto& uvdatum : mset) {
        uvdata.push_back(static_cast<UVDatum<TestType>>(uvdatum));
    }

    std::vector<UVDatum<TestType>> expected(uvdata);

    RaDec original = uvdata.front().meta->phasecenter;

    for (auto& uvdatum : uvdata) {
        phaserotate(uvdatum, {0, 0});
    }

    for (auto& uvdatum : uvdata) {
        uvdatum.meta->phasecenter = {0, 0};
    }

    double maxudiff {}, maxvdiff {}, maxwdiff {}, maxdatadiff {};
    for (size_t i {}; i < uvdata.size(); ++i) {
        maxudiff = std::max<double>(maxudiff, std::abs(uvdata[i].u - expected[i].u));
        maxvdiff = std::max<double>(maxvdiff, std::abs(uvdata[i].v - expected[i].v));
        maxwdiff = std::max<double>(maxwdiff, std::abs(uvdata[i].w - expected[i].w));


        auto data = uvdata[i].data;
        data -= expected[i].data;
        maxdatadiff = std::max<double>(
            maxdatadiff,
            thrust::abs(static_cast<thrust::complex<double>>(data))
        );
    }

    // Ensure we have done some kind of shift
    REQUIRE(maxudiff > 1);
    REQUIRE(maxvdiff > 1);
    REQUIRE(maxwdiff > 1);
    REQUIRE(maxdatadiff > 1);

    for (auto& uvdatum : uvdata) {
        phaserotate(uvdatum, original);
    }

    maxudiff = 0, maxvdiff = 0, maxwdiff = 0, maxdatadiff = 0;
    for (size_t i {}; i < uvdata.size(); ++i) {
        maxudiff = std::max<double>(maxudiff, std::abs(uvdata[i].u - expected[i].u));
        maxvdiff = std::max<double>(maxvdiff, std::abs(uvdata[i].v - expected[i].v));
        maxwdiff = std::max<double>(maxwdiff, std::abs(uvdata[i].w - expected[i].w));


        auto data = uvdata[i].data;
        data -= expected[i].data;
        maxdatadiff = std::max<double>(
            maxdatadiff,
            thrust::abs(static_cast<thrust::complex<double>>(data))
        );
    }

    // Now ensure we have returned back
    double alloweddiff {std::is_same_v<float, TestType> ? 1e-3 : 1e-12};
    REQUIRE(maxudiff < alloweddiff);
    REQUIRE(maxvdiff < alloweddiff);
    REQUIRE(maxwdiff < alloweddiff);
    REQUIRE(maxdatadiff < (std::is_same_v<float, TestType> ? 1e-1 : 1e-9));
    fmt::println("Max data diff: {:g}", maxdatadiff);
}

TEST_CASE("Measurement Set & Partition", "[mset]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx=1000, .imgNy=1000, .imgScalelm=std::sin(deg2rad(15. / 3600)),
        .paddingfactor=1.0, .kernelsize=96, .kernelpadding=18, .wstep=25
    };

    Beam::Uniform<double> beam;
    auto aterm = beam.gridResponse(gridconf.subgrid(), {0, 0}, 0);

    MeasurementSet mset(
        {TESTDATA}, 0, 11, 0, std::numeric_limits<double>::max()
    );

    auto uvdata = mset.data<double>();
    auto workunits = partition(
        uvdata, gridconf, Aterms<double>(aterm)
    );

    size_t n {};
    for (auto& workunit : workunits) {
        n += workunit.data.size();
    }

    REQUIRE( n == mset.size() );

    SECTION("Phase center coordinate conversion") {
        auto radec = mset.phaseCenter();

        double time = mset.midtime();

        auto azel = radecToAzel(radec, time, Beam::MWA<double>::origin);

        // Test results generated from Astropy
        AzEl expected {.az = deg2rad(188.78902259), .el = deg2rad(70.73802277)};

        REQUIRE( std::abs(std::remainder(azel.az - expected.az, 2 * ::pi_v<double>)) < 1e-3 );
        REQUIRE( std::abs(std::remainder(azel.el - expected.el, 2 * ::pi_v<double>)) < 1e-3 );
    }
}

TEST_CASE("Widefield inversion", "[widefield]") {
    using P = float;

    // Config
    double scale_asec = 15;
    const GridConfig gridconf {
        .imgNx = 9000, .imgNy = 9000, .imgScalelm = std::sin(deg2rad(scale_asec / 3600)),
        .paddingfactor = 1.5, .kernelsize = 128, .kernelpadding = 18, .wstep = 25
    };

    const int oversample {6};
    REQUIRE( gridconf.imgNx % oversample == 0 );

    MeasurementSet mset({TESTDATA}, 0, 11);

    // Copy uvdata to vector, for use in both idg and dft
    auto uvdata = mset.data<P>();

    // Create beam
    auto delays = std::get<2>(mset.mwaDelays().front());
    Beam::MWA<P> beam(mset.midtime(), delays);

    auto gridorigin = mset.phaseCenter();

    fmt::println("IDG imaging...");
    auto workunits = partition(
        uvdata,
        gridconf,
        Aterms<P>{beam.gridResponse(gridconf.subgrid(), gridorigin, mset.midfreq())}
    );
    auto img = invert<StokesI, P>(workunits, gridconf);

    fmt::println("Direct DT imaging...");

    auto gridspec = GridSpec::fromScaleLM(
        gridconf.grid().Nx / oversample,
        gridconf.grid().Ny / oversample,
        gridconf.grid().scalelm * oversample
    );
    auto aterm = beam.gridResponse(gridspec, gridorigin, mset.midfreq());

    HostArray<StokesI<P>, 2> expected {gridspec.shape()};
    idft<StokesI, P>(expected, aterm, uvdata, gridspec);

    // Correct for beam power
    for (size_t i {}; i < aterm.size(); ++i) {
        auto a = static_cast<ComplexLinearData<double>>(aterm[i]);
        expected[i] *= static_cast<StokesI<P>>(StokesI<double>::beamPower(a, a));
    }

    HostArray<StokesI<P>, 2> diff {expected};
    for (size_t i {}; i < diff.size(); ++i) {
        auto [xpx, ypx] = gridspec.linearToGrid(i);
        xpx *= oversample; ypx *= oversample;
        auto j = gridconf.grid().gridToLinear(xpx, ypx);
        diff[i] -= img[j];
    }

    P maxdiff {-1};
    for (size_t i {}; i < diff.size(); ++i) {
        maxdiff = std::max(maxdiff, ::abs(diff[i]));
    }
    fmt::println("Max diff: {}", maxdiff);

    REQUIRE(maxdiff < 1e-5);
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
    const GridConfig gridconf {
        .imgNx = 1000, .imgNy = 1000, .imgScalelm = std::sin(deg2rad(2. / 60)),
        .paddingfactor = 1.5, .kernelsize = 96, .kernelpadding = 18, .wstep = 25,
        .deltal = 0.02, .deltam = -0.01
    };
    const GridSpec gridspec = gridconf.grid();
    double freq {150e6};

    // This is not a random coordinate: it is directly overhead
    // the MWA at mjd = 5038236804 / 86400
    RaDec gridorigin(deg2rad(21.5453427), deg2rad(-26.80060483));

    // Create Aterms
    BEAM beam;
    if constexpr(std::is_same<Beam::Gaussian<Q>, BEAM>::value) {
        beam = BEAM(gridorigin, deg2rad(25.));
    }
    if constexpr(std::is_same<Beam::MWA<Q>, BEAM>::value) {
        beam = BEAM(5038236804. / 86400.);
    }

    // Create uvdata
    auto meta = makesharedhost<UVMeta>(0, 0, 0, 0, RaDec{0, 0});
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
                beam.pointResponse(lmToRaDec(l, m, gridorigin), freq)
            );
            sources.emplace_back(l, m, jones);
        }

        for (size_t i {}; i < 20000; ++i) {
            double u = rand(gen), v = rand(gen), w = rand(gen);

            u = (u - 0.5) * 200;
            v = (v - 0.5) * 200;
            w = (w - 0.5) * 200;

            ComplexLinearData<double> data;
            for (auto [l, m, jones] : sources) {
                auto phase = cispi(-2 * (
                    u * l + v * m + w * ndash(l, m)
                ));

                ComplexLinearData<double> cell {phase, 0, 0, phase};
                data += matmul(matmul(jones, cell), jones.adjoint());
            }

            LinearData<double> weights {
                rand(gen), rand(gen), rand(gen), rand(gen)
            };

            // We create UVDatum<P> but avoid using the constructor so that we can test
            // forcing w > 0.
            UVDatum<double> uvdatum;
            uvdatum.meta = meta;
            uvdatum.chan = i;
            uvdatum.u = u; uvdatum.v = v; uvdatum.w = w;
            uvdatum.weights = weights;
            uvdatum.data = data;

            uvdata64.push_back(uvdatum);
        }
    }

    // Weight naturally
    const Natural<double> weighter(uvdata64, gridconf.padded());
    applyWeights(weighter, uvdata64);

    // Calculate expected at double precision
    HostArray<StokesI<double>, 2> expected {gridspec.shape()};
    {
        auto jones = static_cast<HostArray<ComplexLinearData<double>, 2>>(
            beam.gridResponse(gridspec, gridorigin, freq)
        );
        idft<StokesI, double>(expected, jones, uvdata64, gridspec);
    }

    // Cast to float or double AND set w >= 0
    std::vector<UVDatum<Q>> uvdata(uvdata64.size());
    for (size_t i {}; const auto& uvdatum : uvdata64) {
        uvdata[i++] = static_cast<UVDatum<Q>>(uvdatum).forcePositiveW();
    }

    auto Aterm = beam.gridResponse(gridconf.subgrid(), gridorigin, freq);
    auto workunits = partition(uvdata, gridconf, Aterms<Q>(Aterm));
    auto img = invert<StokesI, Q>(workunits, gridconf);

    // Correct for beam
    auto jonesgrid = beam.gridResponse(gridspec, gridorigin, freq);
    for (size_t i {}; i < gridspec.size(); ++i) {
        img[i] /= StokesI<Q>::beamPower(jonesgrid[i], jonesgrid[i]);
    }

    double maxdiff {-1};
    for (size_t i {}; i < gridspec.size(); ++i) {
        double diff = thrust::abs(
            expected[i].I.real() - thrust::complex<double>(img[i].I.real())
        );
        maxdiff = std::max(maxdiff, diff);
    }
    fmt::println("Max diff: {:g}", maxdiff);

    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < THRESHOLDF * std::pow(10, THRESHOLDP));
}

TEMPLATE_TEST_CASE_SIG(
    "Predict", "[predict]",
    ((typename Q, typename BEAM, int THRESHOLDF, int THRESHOLDP), Q, BEAM, THRESHOLDF, THRESHOLDP),
    (float, (UniformBeam<float>), 3, -5),
    (double, (UniformBeam<double>), 2, -12),
    (float, (GaussianBeam<float>), 3, -5),
    (double, (GaussianBeam<double>), 2, -12),
    (float, (MWABeam<float>), 4, -5),
    (double, (MWABeam<double>), 3, -5)
) {
    const GridConfig gridconf {
        .imgNx = 2000, .imgNy = 2000, .imgScalelm = 1. / (4000 * 20), .paddingfactor = 2,
        .kernelsize = 96, .kernelpadding = 17, .wstep = 25, .deltal = 0.015, .deltam = -0.01
    };
    const GridSpec gridspec = gridconf.grid();
    const double freq {150e6};

    // This is not a random coordinate: it is directly overhead
    // the MWA at mjd = 5038236804 / 86400
    const RaDec phaseCenter(deg2rad(21.5453427), deg2rad(-26.80060483));

    BEAM beam;
    if constexpr(std::is_same_v<BEAM, GaussianBeam<Q>>) {
        beam = BEAM(phaseCenter, deg2rad(25.));
    }
    if constexpr(std::is_same_v<BEAM, MWABeam<Q>>) {
        beam = BEAM(5038236804. / 86400.);
    }

    // Create skymap
    HostArray<StokesI<Q>, 2> skymap {gridspec.shape()};

    std::mt19937 gen(1234);
    std::uniform_int_distribution<int> randints(0, gridspec.size());

    for (size_t i {}; i < 1000; ++i) {
        skymap[randints(gen)] = StokesI<Q> {Q(1)};
    }

    std::uniform_real_distribution<Q> randfloats(-1, 1);

    // Create empty UVDatum
    std::vector<UVDatum<Q>> uvdata;
    auto meta = makesharedhost<UVMeta>(0, 12345.6, 1, 5, RaDec{0, 1});
    for (size_t i {}; i < 5000; ++i) {
        Q u {randfloats(gen)}, v {randfloats(gen)}, w {randfloats(gen)};
        u = u * 1000;
        v = v * 1000;
        w = w * 500;

        // TODO: use emplace_back() when we can upgrade Clang
        uvdata.push_back({
            meta, static_cast<int>(i), u, v, w,
            LinearData<Q> {1, 1, 1, 1},
            ComplexLinearData<Q> {0, 0, 0, 0}
        });
    }

    // Weight naturally
    Natural<Q> weighter(uvdata, gridconf.padded());
    applyWeights(weighter, uvdata);

    // Calculate expected at double precision
    std::vector<UVDatum<double>> expected;
    {
        // Find non-empty pixels
        // TODO: replace with copy_if and std::back_inserter
        std::vector<size_t> idxs;
        for (size_t i {}; i < skymap.size(); ++i) {
            if (thrust::abs(skymap[i].I) != 0) idxs.push_back(i);
        }

        // Calculate and cache beam point responses
        std::unordered_map<size_t, ComplexLinearData<double>> Aterms;
        for (auto idx : idxs) {
            auto [l, m] = gridspec.linearToSky<double>(idx);
            auto radec = lmToRaDec(l, m, phaseCenter);
            Aterms[idx] = static_cast<ComplexLinearData<double>>(
                beam.pointResponse(radec, freq)
            );
        }

        // For each UVDatum, sum over non-empty pixels
        for (const auto& uvdatum : uvdata) {
            UVDatum<double> uvdatum64 = static_cast<UVDatum<double>>(uvdatum);
            for (auto idx : idxs) {
                // Convert StokesI<Q> -> StokesI<double> -> ComplexLinearData<double>
                ComplexLinearData<double> cell = StokesI<double>(skymap[idx].I);

                // Apply beam attenuation
                auto Aterm = Aterms[idx];
                cell = matmul(matmul(Aterm, cell), Aterm.adjoint());

                // Apply phase
                auto [l, m] = gridspec.linearToSky<double>(idx);
                cell *= cispi(
                    -2 * (uvdatum64.u * l + uvdatum64.v * m + uvdatum64.w * ndash(l, m))
                );

                uvdatum64.data += cell;
            }
            expected.push_back(uvdatum64);
        }
    }

    // Predict using IDG
    auto aterm = beam.gridResponse(gridconf.subgrid(), phaseCenter, freq);

    auto workunits = partition(
        uvdata, gridconf, Aterms<Q>(aterm)
    );

    predict<StokesI<Q>, Q>(
        workunits, skymap, gridconf, DegridOp::Replace
    );

    // Flatten workunits back into uvdata and sort back to original order
    // using the chan attribute
    for (const auto& workunit : workunits) {
        for (const auto uvdatumptr : workunit.data) {
            uvdata[uvdatumptr->chan] = *uvdatumptr;
        }
    }

    // Create images to compare diff
    auto fullAterms = beam.gridResponse(gridspec, phaseCenter, freq);

    HostArray<StokesI<Q>, 2> imgMap {gridspec.shape()};
    idft<StokesI, Q>(imgMap, fullAterms, uvdata, gridspec);

    HostArray<StokesI<double>, 2> expectedMap {gridspec.shape()};
    idft<StokesI, double>(
        expectedMap, static_cast<HostArray<ComplexLinearData<double>, 2>>(fullAterms),
        expected, gridspec
    );

    double maxdiff {-1};
    for (size_t i {}; i < gridspec.size(); ++i) {
        auto diff = expectedMap[i].I - imgMap[i].I;

        maxdiff = std::max<double>(
            maxdiff, thrust::abs(diff)
        );
    }

    fmt::println("Prediction max diff: {}", maxdiff);

    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < THRESHOLDF * std::pow(10, THRESHOLDP) );
    REQUIRE( uvdata[0].meta->time == 12345.6 );
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

    std::vector<MeasurementSet::FreqRange> freqs;
    std::vector<std::vector<HostArray<StokesI<double>, 2>>> residualss(1);
    std::vector<std::vector<HostArray<thrust::complex<double>, 2>>> psfss(1);

    auto& residuals = residualss.front();
    auto& psfs = psfss.front();

    // Convolve the models and populate each of input vectors
    for (int n {}; n < N; ++n) {
        auto expected = convolve(models[n], dirtyPSF);
        expectedSum += expected;

        freqs.push_back({1e6 * (n + 1), 1e6 * (n + 2)});
        psfs.emplace_back(dirtyPSF);
        residuals.push_back(std::move(expected));
    }
    expectedSum /= StokesI<double>(N);

    // Find peak max, which we use as the final test value
    double maxInit {};
    for (size_t i {}; i < expectedSum.size(); ++i) {
        maxInit = std::max(maxInit, std::abs(expectedSum[i].I.real()));
    };

    auto [components, iter, _] = clean::major(
        freqs, residualss, {gridspec}, psfss,
        0.01, 0.991, 0, 0, std::numeric_limits<size_t>::max()
    );

    auto fittedPSF = PSF<double>(dirtyPSF, gridspec).draw(gridspec);

    // Combine the convolved, component images to produce a restored image
    HostArray<StokesI<double>, 2> restoredSum {gridspec.Nx, gridspec.Ny};
    for (auto& component : components) {
        HostArray<StokesI<double>, 2> componentMap {gridspec.Nx, gridspec.Ny};
        for (auto& [lmpx, val] : component) {
            auto [lpx, mpx] = lmpx;
            auto idx = gridspec.LMpxToLinear(lpx, mpx);
            if (idx) componentMap[*idx] += val;
        }

        restoredSum += convolve(componentMap, fittedPSF);
    }
    restoredSum /= StokesI<double>(N);

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