#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "aterms.h"
#include "beam.h"
#include "clean.h"
#include "config.h"
#include "datatable.h"
#include "dft.h"
#include "fits.h"
#include "invert.h"
#include "logger.h"
#include "memory.h"
#include "psf.h"
#include "partition.h"
#include "predict.h"
#include "taper.h"
#include "util.h"
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

    // Device matmul has a special FMA implementation
    {
        DeviceArray<ComplexLinearData<double>, 1> C_d(1);
        map([A = A, B = B] __device__ (auto& C) {
            C = matmul(A, B);
        }, C_d);
        REQUIRE(HostArray<ComplexLinearData<double>, 1>(C_d)[0] == matmul(A, B));
    }

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

TEST_CASE("FFT and central shifts", "[fft]") {
    for (auto Nx : {1022, 1024, 1026, 1028}) {
        for (auto Ny : {1022, 1024, 1026, 1028}) {
            GridSpec gridspec {.Nx=Nx, .Ny=Ny};

            HostArray<thrust::complex<double>, 2> arr(gridspec.Ny, gridspec.Nx);
            arr[gridspec.gridToLinear(gridspec.Nx / 2, gridspec.Ny / 2)] = 1;
            DeviceArray<thrust::complex<double>, 2> arr_d {arr};

            auto plan = fftPlan<thrust::complex<double>>(gridspec);
            fftExec(plan, arr_d, HIPFFT_FORWARD);
            copy(arr, arr_d);
            hipfftDestroy(plan);

            REQUIRE( std::all_of(arr.begin(), arr.end(), [] (auto x) {
                return std::abs(x.real() - 1) < 1e-14 && std::abs(x.imag() - 0) < 1e-14;
            }) );
        }
    }
}

TEST_CASE("Toml configuration", "[toml]") {
    // For now, we just test that:
    // 1. the config object can be converted to toml
    // 2. the toml can be converted back to Config
    // 3. and all parameter values are retained

    Config config1 {
        .loglevel = Logger::Level::debug, .datacolumn = DataTable::DataColumn::corrected,
        .chanlow = 33, .chanhigh = 56, .channelsOut = 5,
        .msets = {"/path1.fits", "/path2.fits"}, .maxDuration = 32,
        .weight = "briggs", .robust = 1.3,
        .scale = 25, .phasecenter = RaDec{0.5, 0.5},
        .fields = {{.Nx = 1234, .Ny = 4568, .projectioncenter = RaDec{0.5, 0.5}, .phasecorrections = {"/phases1.fits", "/phases2.fits"}}},
        .precision = 64, .kernelsize = 156, .kernelpadding = 9, .paddingfactor = 1.23,
        .gpumem = 25.6, .majorgain = 0.354, .minorgain = 0.2343,
        .cleanThreshold = 543154, .autoThreshold = 431.54, .nMajor = 432, .nMinor = 5426543,
        .spectralparams = 123
    };

    auto configtext = std::istringstream(toml::format(
        toml::basic_value<toml::preserve_comments>(config1)
    ));
    Config config2 = toml::expect<Config>(toml::parse(configtext)).unwrap();

    REQUIRE(config1 == config2);
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

    auto gridspec = GridSpec::fromScaleLM(N, N, scalelm);
    auto [l0, m0] = gridspec.gridToLM<double>(lpx, mpx);

    auto radec = lmToRaDec(l0, m0, gridorigin);
    {
        // Calculcated using Astropy
        RaDec expected {.ra=deg2rad(20.12114593), .dec=deg2rad(-39.48407916)};

        REQUIRE(std::abs(radec.ra - expected.ra) < 1e-5);
        REQUIRE(std::abs(radec.dec - expected.dec) < 1e-5);
    }

    auto [l1, m1] = RaDecTolm(radec, gridorigin);
    REQUIRE(std::abs(l0 - l1) < 1e-9);
    REQUIRE(std::abs(m0 - m1) < 1e-9);
}

TEST_CASE("GridSpec", "[gridspec]") {
    auto gridspec = GridSpec::fromScaleLM(1000, 1200, 0.01);

    bool ok {true};
    for (size_t idx {}; idx < gridspec.size(); ++idx) {
        auto [xpx, ypx] = gridspec.linearToGrid(idx);
        ok &= 0 <= xpx && xpx < 1000 && 0 <= ypx && ypx < 1200;
        ok &= idx == gridspec.gridToLinear(xpx, ypx);

        auto [u, v] = gridspec.gridToUV<double>(xpx, ypx);
        auto [upx, vpx] = gridspec.UVtoGrid<double>(u, v);
        ok &= (upx - xpx) < 1e-10 && (vpx - ypx) < 1e-10;
    }

    REQUIRE(ok);
}

TEST_CASE("Utility functions", "[utility]") {
    auto smallGridspec = GridSpec::fromScaleUV(128, 128, 1, 1);
    auto largeGridspec = GridSpec::fromScaleUV(1000, 1000, 1, 1);

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

TEST_CASE("Phase rotation", "[phaserotation]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    DataTable tbl(TESTDATA, {.chanlow=0, .chanhigh=12});
    DataTable expected = tbl;

    RaDec original = tbl.phasecenter();

    tbl.phasecenter({0, 0});

    // Ensure something has changed
    REQUIRE(tbl.metadata(12345).u != expected.metadata(12345).u);
    REQUIRE(tbl.metadata(12345).v != expected.metadata(12345).v);
    REQUIRE(tbl.metadata(12345).w != expected.metadata(12345).w);

    tbl.phasecenter(original);

    double maxudiff = 0, maxvdiff = 0, maxwdiff = 0, maxdatadiff = 0;
    for (size_t irow {}; irow < tbl.nrows(); ++irow) {
        auto m1 = tbl.metadata(irow);
        auto m2 = expected.metadata(irow);

        maxudiff = std::max<double>(maxudiff, std::abs(m1.u - m2.u));
        maxvdiff = std::max<double>(maxvdiff, std::abs(m1.v - m2.v));
        maxwdiff = std::max<double>(maxwdiff, std::abs(m1.w - m2.w));

        for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
            ComplexLinearData<float> datum = tbl.data(irow, ichan);
            datum -= expected.data(irow, ichan);

            maxdatadiff = std::max<double>(
                maxdatadiff,
                thrust::abs(static_cast<thrust::complex<double>>(datum))
            );
        }
    }

    // Now ensure we have returned back
    double alloweddiff {1e-7};
    REQUIRE(maxudiff < alloweddiff);
    REQUIRE(maxvdiff < alloweddiff);
    REQUIRE(maxwdiff < alloweddiff);
    REQUIRE(maxdatadiff < 2e-3);
    fmt::println("Max uvw diffs: {:g}, {:g}, {:g}", maxudiff, maxvdiff, maxwdiff);
    fmt::println("Max data diff: {:g}", maxdatadiff);
}

TEST_CASE("Measurement Set & Partition", "[mset]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    GridConfig gridconf {
        .imgNx=1000, .imgNy=1000, .imgScalelm=std::sin(deg2rad(15. / 3600)),
        .paddingfactor=1.0, .kernelsize=96, .kernelpadding=18,
    };

    DataTable tbl(TESTDATA, {});
    auto beam = Beam::Uniform<double>().gridResponse(gridconf.subgrid(), {}, 0);
    Aterms::StaticCorrections aterms(beam);
    auto workunits = partition(tbl, gridconf, aterms);

    size_t n {};
    for (auto& workunit : workunits) {
        n += workunit.size();
    }

    REQUIRE( n == tbl.size() );

    // Test whether each data point actually lies in its subgrid
    const auto subgridspec = gridconf.subgrid();
    const auto gridspec = gridconf.padded();
    const auto lambdas = tbl.lambdas();
    const double radius = subgridspec.Nx / 2 - gridconf.kernelpadding + 0.5;

    for (auto workunit : workunits) {
        // Subtract 0.5 to center pixel
        double u0px = workunit.upx - 0.5;
        double v0px = workunit.vpx - 0.5;

        for (size_t irow {workunit.rowstart}; irow < workunit.rowend; ++irow) {
            auto m =  tbl.metadata(irow);

            // REQUIRE is very slow; first check the loop for a failure
            bool rowok {true};
            for (size_t ichan {workunit.chanstart}; ichan < workunit.chanend; ++ichan) {
                double u = m.u / lambdas[ichan];
                double v = m.v / lambdas[ichan];

                // We force w > 0 during paritition, thanks to V(u,v,w) = V(-u,-v-w)^H
                if (m.w / lambdas[ichan] < 0) { u *= -1; v *= -1; }

                auto [upx, vpx] = gridspec.UVtoGrid<double>(u, v);
                rowok = rowok && std::abs(upx - u0px) < radius;
                rowok = rowok && std::abs(vpx - v0px) < radius;
            }

            if (!rowok) {
                for (size_t ichan {workunit.chanstart}; ichan < workunit.chanend; ++ichan) {
                    double u = m.u / lambdas[ichan];
                    double v = m.v / lambdas[ichan];

                    // We force w > 0 during paritition, thanks to V(u,v,w) = V(-u,-v-w)^H
                    if (m.w / lambdas[ichan] < 0) { u *= -1; v *= -1; }

                    auto [upx, vpx] = gridspec.UVtoGrid<double>(u, v);
                    REQUIRE(std::abs(upx - u0px) < radius);
                    REQUIRE(std::abs(vpx - v0px) < radius);
                }
            }

            // Check that pixel coordinates and u,v values match
            auto [u, v] = gridspec.gridToUV<double>(
                workunit.upx, workunit.vpx
            );

            REQUIRE(std::abs(u - workunit.u) < 1e-8);
            REQUIRE(std::abs(v - workunit.v) < 1e-8);
        }
    }

    SECTION("Phase center coordinate conversion") {
        auto radec = tbl.phasecenter();

        double time = tbl.midtime();

        auto azel = radecToAzel(radec, time, Beam::MWA<double>::origin);

        // Test results generated from Astropy
        AzEl expected {.az = deg2rad(188.78902259), .el = deg2rad(70.73802277)};

        REQUIRE( std::abs(std::remainder(azel.az - expected.az, 2 * ::pi_v<double>)) < 1e-3 );
        REQUIRE( std::abs(std::remainder(azel.el - expected.el, 2 * ::pi_v<double>)) < 1e-3 );
    }
}

TEST_CASE("Widefield inversion", "[widefield]") {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    using P = float;

    // Config
    double scale_asec = 15;
    const GridConfig gridconf {
        .imgNx = 12000, .imgNy = 12000, .imgScalelm = std::sin(deg2rad(scale_asec / 3600)),
        .paddingfactor = 1.5, .kernelsize = 128, .kernelpadding = 18,
    };

    const int oversample {16};
    REQUIRE( gridconf.imgNx % oversample == 0 );

    DataTable tbl(TESTDATA, {.chanlow=0, .chanhigh=11});

    // Weight the data
    Natural weighter(tbl, gridconf.padded());
    applyweights(weighter, tbl);

    // Create aterms
    Aterms::BeamCorrections<P> aterms(
        {casacore::MeasurementSet(TESTDATA)}, gridconf.subgrid(),
        99999999, tbl.phasecenter(), tbl.midfreq()
    );

    fmt::println("IDG imaging...");
    auto workunits = partition(tbl, gridconf, aterms);
    auto img = invert<StokesI, P>(tbl, workunits, gridconf, aterms);

    fmt::println("Direct DT imaging...");
    auto gridspec = GridSpec::fromScaleLM(
        gridconf.imgNx / oversample,
        gridconf.imgNy / oversample,
        gridconf.imgScalelm * oversample
    );

    aterms = Aterms::BeamCorrections<P>(
        {casacore::MeasurementSet(TESTDATA)}, gridspec,
        99999999, tbl.phasecenter(), tbl.midfreq()
    );

    HostArray<StokesI<P>, 2> expected {gridspec.shape()};
    auto jones = static_cast<HostArray<ComplexLinearData<P>, 2>>(
        *std::get<1>(aterms.get(tbl.midtime(), 0))
    );
    idft<StokesI, P>(expected, tbl, jones, gridspec, true);

    HostArray<StokesI<P>, 2> diff {expected};
    for (size_t i {}; i < diff.size(); ++i) {
        auto [xpx, ypx] = gridspec.linearToGrid(i);
        xpx *= oversample; ypx *= oversample;
        auto j = gridconf.grid().gridToLinear(xpx, ypx);
        diff[i] -= img[j];
    }

    fits::save("image.fits", img, gridconf.grid(), tbl.phasecenter());
    fits::save("expected.fits", expected, gridspec, tbl.phasecenter());
    fits::save("diff.fits", diff, gridspec, tbl.phasecenter());

    P maxdiff {-1};
    for (size_t i {}; i < diff.size(); ++i) {
        maxdiff = std::max(maxdiff, ::abs(diff[i].I.real()));
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
    (float, (UniformBeam<double>), 3, -5),
    (double, (UniformBeam<double>), 5, -10),
    (float, (GaussianBeam<double>), 3, -5),
    (double, (GaussianBeam<double>), 5, -10),
    (float, (MWABeam<double>), 3, -5),
    (double, (MWABeam<double>), 2, -8)
) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    // Config
    const GridConfig gridconf {
        .imgNx = 1000, .imgNy = 1050, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .paddingfactor = 1.5, .kernelsize = 96, .kernelpadding = 18,
        .deltal = 0.02, .deltam = -0.01
    };
    const GridSpec gridspec = gridconf.grid();
    double freq {150e6};

    // This is not a random coordinate: it is directly overhead
    // the MWA at mjd = 5038236804 / 86400
    RaDec gridorigin(deg2rad(21.5453427), deg2rad(-26.80060483));

    // Create Aterms
    BEAM beam;
    if constexpr(std::is_same<Beam::Gaussian<double>, BEAM>::value) {
        beam = BEAM(gridorigin, deg2rad(25.));
    }
    if constexpr(std::is_same<Beam::MWA<double>, BEAM>::value) {
        beam = BEAM(5038236804. / 86400.);
    }

    DataTable tbl(TESTDATA, {.chanlow=24, .chanhigh=26});

    // Weight naturally
    const Natural weighter(tbl, gridconf.padded());
    applyweights(weighter, tbl);

    // Calculate expected at double precision
    HostArray<StokesI<double>, 2> expected {gridspec.shape()};
    {
        auto jones = beam.gridResponse(gridspec, gridorigin, freq);
        idft<StokesI, double>(expected, tbl, jones, gridspec, true);
    }
    // fits::save("expected.fits", expected, gridspec, {0, 0});

    // Now apply phase shift to visibilities
    // Each let theta = 0.1 x antid
    double midtime = tbl.midtime();
    for (size_t irow {}; irow < tbl.nrows(); ++irow) {
        auto m = tbl.metadata(irow);
        double f = m.time < midtime ? 0.1 : -0.3;
        thrust::complex<float> theta = cispi(f * (m.baseline.a - m.baseline.b));
        for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
            tbl.data(irow, ichan) *= theta;
        }
    }

    // Construct A terms
    auto aterms = [&] {
        // First: make phase corrections
        std::vector<Interval> intervals {
            Interval{0, midtime},
            Interval{midtime, std::numeric_limits<double>::infinity()}
        };
        HostArray<Q, 4> phases(
            std::array<long long, 4>{2, 128, gridconf.kernelsize, gridconf.kernelsize}
        );
        for (int antid {}; antid < 128; ++antid) {
            phases(0)(antid).fill(0.1 * ::pi_v<double> * antid);
            phases(1)(antid).fill(-0.3 * ::pi_v<double> * antid);
        }
        auto phasecorrections = std::make_shared<Aterms::PhaseCorrections<Q>>(
            intervals, phases
        );

        // Then: beam corrections
        auto beamcorrections = std::make_shared<Aterms::StaticCorrections<Q>>(
            HostArray<ComplexLinearData<Q>, 2>(
                beam.gridResponse(gridconf.subgrid(), gridorigin, freq)
            )
        );

        // Finally: merge into combined corrections object
        return Aterms::CombinedCorrections<Q>(
            std::static_pointer_cast<Aterms::Interface<Q>>(beamcorrections),
            std::static_pointer_cast<Aterms::Interface<Q>>(phasecorrections)
        );
    }();

    auto workunits = partition(tbl, gridconf, aterms);
    auto img = invert<StokesI, Q>(tbl, workunits, gridconf, aterms);
    // fits::save("image.fits", img, gridspec, {0, 0});

    double rmsexpected {};
    HostArray<double, 2> diff(img.shape());
    for (size_t i {}; auto& px : diff) {
        px = expected[i].I.real();
        px -= img[i].I.real();
        ++i;

        rmsexpected += expected[i].I.real() * expected[i].I.real();
    }
    // fits::save("diff.fits", diff, gridspec, {0, 0});

    rmsexpected = std::sqrt(rmsexpected / gridspec.size());
    fmt::println("Original RMS: {:g}", rmsexpected);

    double maxdiff {-1};
    double rms {};
    for (size_t i {}; i < gridspec.size(); ++i) {
        double diff = thrust::abs(
            expected[i].I.real() - thrust::complex<double>(img[i].I.real())
        );
        rms += diff * diff;
        maxdiff = std::max(maxdiff, diff);
    }
    rms = std::sqrt(rms / gridspec.size());
    fmt::println("Max diff: {:g}", maxdiff);
    fmt::println("RMS error: {:g}", rms);

    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < THRESHOLDF * std::pow(10, THRESHOLDP));
    REQUIRE( rms < 0.1 * THRESHOLDF * std::pow(10, THRESHOLDP));
}

TEMPLATE_TEST_CASE_SIG(
    "Predict", "[predict]",
    ((typename Q, typename BEAM, int THRESHOLDF, int THRESHOLDP), Q, BEAM, THRESHOLDF, THRESHOLDP),
    (float, (UniformBeam<double>), 5, -5),
    (double, (UniformBeam<double>), 1, -9),
    (float, (GaussianBeam<double>), 5, -5),
    (double, (GaussianBeam<double>), 8, -10),
    (float, (MWABeam<double>), 5, -5),
    (double, (MWABeam<double>), 8, -10)
) {
    if (!TESTDATA) { SKIP("TESTDATA path not provided"); }

    const GridConfig gridconf {
        .imgNx = 2000, .imgNy = 1800, .imgScalelm = std::sin(deg2rad(15. / 3600)),
        .paddingfactor = 1.5, .kernelsize = 96, .kernelpadding = 17,
        .deltal = 0.015, .deltam = -0.01
    };
    const GridSpec gridspec = gridconf.grid();
    const double freq {150e6};

    // This is not a random coordinate: it is directly overhead
    // the MWA at mjd = 5038236804 / 86400
    const RaDec phaseCenter(deg2rad(21.5453427), deg2rad(-26.80060483));

    BEAM beam;
    if constexpr(std::is_same_v<BEAM, GaussianBeam<double>>) {
        beam = BEAM(phaseCenter, deg2rad(5.));
    }
    if constexpr(std::is_same_v<BEAM, MWABeam<double>>) {
        beam = BEAM(5038236804. / 86400.);
    }

    // Create skymap
    HostArray<StokesI<Q>, 2> skymap {gridspec.shape()};

    std::mt19937 gen(1234);
    std::uniform_int_distribution<int> randints(0, gridspec.size());

    for (size_t i {}; i < 10; ++i) {
        skymap[randints(gen)] = StokesI<Q> {Q(1)};
    }

    // Create zeroed DataTable
    DataTable tbl(TESTDATA, {.chanlow=24, .chanhigh=26});
    for (auto& datum : tbl.data()) datum = {0, 0, 0, 0};
    for (auto& weight : tbl.weights()) weight = {1, 1, 1, 1};

    // Weight naturally
    Natural weighter(tbl, gridconf.padded());
    applyweights(weighter, tbl);

    // Calculate expected at double precision
    DataTable expectedtbl(tbl);

    // Predict using IDG
    {
        Aterms::StaticCorrections aterms(HostArray<ComplexLinearData<Q>, 2>(
            beam.gridResponse(gridconf.subgrid(), phaseCenter, freq)
        ));
        auto workunits = partition(tbl, gridconf, aterms);

        predict<StokesI, Q>(
            tbl, workunits, skymap, gridconf, aterms, DegridOp::Add
        );
    }

    // For each non-empty pixel, add to each element of data
    for (size_t i {}; i < skymap.size(); ++i) {
        if (skymap[i].I == 0) continue;

        // Convert StokesI<Q> -> StokesI<double> -> ComplexLinearData<double>
        ComplexLinearData<double> cell = static_cast<StokesI<double>>(skymap[i]);

        auto [l, m] = gridspec.linearToSky<double>(i);
        double n {ndash(l, m)};

        // Apply beam attenuation
        auto radec = lmToRaDec(l, m, phaseCenter);
        auto Aterm = beam.pointResponse(radec, freq);
        cell = matmul(matmul(Aterm, cell), Aterm.adjoint());

        for (size_t irow {}; irow < expectedtbl.nrows(); ++irow) {
            for (size_t ichan {}; ichan < expectedtbl.nchans(); ++ichan) {
                auto [u, v, w] = expectedtbl.uvw(irow, ichan);

                ComplexLinearData<double> datum(cell);
                datum *= cispi(2 * (u * l + v * m + w * n));
                expectedtbl.data(irow, ichan) += datum;
            }
        }
    }

    // Create images to compare diff
    HostArray<StokesI<Q>, 2> imgMap {gridspec.shape()};
    idft<StokesI, Q>(
        imgMap, tbl,
        Beam::Uniform<Q>().gridResponse(gridspec, phaseCenter, freq),
        gridspec
    );
    // fits::save("image.fits", imgMap, gridspec, {0, 0});

    HostArray<StokesI<double>, 2> expectedMap {gridspec.shape()};
    idft<StokesI, double>(
        expectedMap, expectedtbl,
        Beam::Uniform<double>().gridResponse(gridspec, phaseCenter, freq),
        gridspec
    );
    // fits::save("expected.fits", expectedMap, gridspec, {0, 0});

    double maxdiff {-1};
    for (size_t i {}; i < gridspec.size(); ++i) {
        auto diff = thrust::abs(expectedMap[i].I - imgMap[i].I);
        maxdiff = std::max<double>(maxdiff, diff);
    }

    fmt::println("Prediction max diff: {}", maxdiff);

    REQUIRE( maxdiff != -1 );
    REQUIRE( maxdiff < THRESHOLDF * std::pow(10, THRESHOLDP) );
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

    std::vector<DataTable::FreqRange> freqs;
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
        0.01, 0.991, 0, 0, std::numeric_limits<size_t>::max(), 2
    );

    auto fittedPSF = PSF<double>(dirtyPSF, gridspec).draw(gridspec);

    // Combine the convolved, component images to produce a restored image
    HostArray<StokesI<double>, 2> restoredSum {gridspec.Ny, gridspec.Nx};
    for (auto& component : components) {
        HostArray<StokesI<double>, 2> componentMap {gridspec.Ny, gridspec.Nx};
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