#pragma once

#include <algorithm>
#include <boost/mpi.hpp>
#include <generator>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>

#include "beam.h"
#include "clean.h"
#include "config.h"
#include "fits.h"
#include "gridspec.h"
#include "invert.h"
#include "mpi.h"
#include "mset.h"
#include "predict.h"
#include "psf.h"
#include "taper.h"
#include "workunit.h"
#include "util.h"
#include "weighter.h"

namespace routines {

template <typename P>
void cleanQueen(const Config& config, boost::mpi::intercommunicator hive) {
    fmt::println("I am the colony queen! All shall love me and despair.");

    const int hivesize = hive.remote_size();
    fmt::println("Hive size is: {}", hivesize);

    const std::ranges::iota_view<int, int> srcs{0, hivesize};

    std::vector<double> freqs(hivesize);
    for (const int src : srcs) {
        hive.recv(src, boost::mpi::any_source, freqs[src]);
    }

    std::vector<HostArray<thrust::complex<P>, 2>> psfs(hivesize);
    for (const int src : srcs) {
        hive.recv(src, boost::mpi::any_source, psfs[src]);
    }

    PSF<P> psf;
    GridSpec gridspecPsf;
    {
        // Collect psfs from workers
        HostArray<thrust::complex<P>, 2> psfCombined {config.gridconf.grid().shape()};
        for (const auto& psf : psfs) {
            psfCombined += psf;
        }
        psfCombined /= thrust::complex<P>(hivesize);
        save("psf.fits", psfCombined);

        // Further crop psfs so that all pixels > 0.2 * (1 - majorgain)
        // are included in cutout. We use this smaller window to speed up PSF fitting and
        // cleaning.
        gridspecPsf = cropPsf(
            psfCombined, config.gridconf.grid(), 0.2 * (1 - config.majorgain)
        );

        // Send gridspecPsf to workers; they will use this to crop their own psfs
        for (const int src : srcs) {
            hive.send(src, 0, gridspecPsf);
        }

        for (auto& psf : psfs) {
            psf = resize(psf, config.gridconf.grid(), gridspecPsf);
        }
        psfCombined = resize(psfCombined, config.gridconf.grid(), gridspecPsf);

        // Fit the combined psf to construct Gaussian PSF
        psf = PSF(psfCombined, gridspecPsf);
    }

    // Collect initial residual images from workers
    std::vector<HostArray<StokesI<P>, 2>> residuals(hivesize);
    for (const int src : srcs) {
        hive.recv(src, boost::mpi::any_tag, residuals[src]);
    }

    // Write out dirty combined
    {
        HostArray<StokesI<P>, 2> dirtyCombined {config.gridconf.grid().shape()};
        for (auto& residual : residuals) {
            dirtyCombined += residual;
        }

        save("dirty.fits", dirtyCombined);
    }

    // Major clean loops
    for (
        size_t imajor {}, iminor {};
        imajor < config.nMajor && iminor < config.nMinor;
        ++imajor
    ) {
        // Signal continuation of clean loop to workers
        for (const int src : srcs) { hive.send(src, 0, true); }

        auto [minorComponentsMaps, iters, finalMajor] = clean::major<P>(
            freqs, residuals, config.gridconf.grid(), psfs, gridspecPsf,
            config.minorgain, config.majorgain, config.cleanThreshold,
            config.autoThreshold, config.nMinor - iminor
        );
        iminor += iters;

        // Send out clean components for subtraction from visibilities by workers
        for (const int src : srcs) {
            HostArray<StokesI<P>, 2> minorComponents {config.gridconf.grid().shape()};
            for (auto [idx, val] : minorComponentsMaps[src]) {
                minorComponents[idx] = val;
            }

            hive.send(src, 0, minorComponents);
        }

        // Gather new residuals from workers
        for (const int src : srcs) {
            hive.recv(src, boost::mpi::any_tag, residuals[src]);
        }

        if (finalMajor) {
            break;
        }
    }

    // Signal end of clean loop to workers
    for (const int src : srcs) { hive.send(src, 0, false); }

    // Write out the final images to disk
    HostArray<StokesI<P>, 2> residualCombined {config.gridconf.grid().shape()};
    HostArray<StokesI<P>, 2> componentsCombined {config.gridconf.grid().shape()};

    {
        HostArray<StokesI<P>, 2> components;
        for (const int src : srcs) {
            residualCombined += residuals[src];
            hive.recv(src, boost::mpi::any_tag, components);
            componentsCombined += components;
        }
    }

    residualCombined /= StokesI<P>(hivesize);
    componentsCombined /= StokesI<P>(hivesize);

    auto imageCombined = convolve(componentsCombined, psf.draw(config.gridconf.grid()));
    imageCombined += residualCombined;

    save("residual.fits", residualCombined);
    save("components.fits", componentsCombined);
    save("image.fits", imageCombined);
}

template <typename P>
void cleanWorker(
    const Config& config,
    boost::mpi::intercommunicator queen,
    boost::mpi::communicator hive
) {
    const int rank = hive.rank();
    const int hivesize = hive.size();

    fmt::println("Worker bee with rank {}, reporting for duty!", rank);

    // Be careful in these channel boundary calculations to remember that the
    // upper channel is inclusive in the range
    const int chanwidth = (config.chanhigh - config.chanlow + 1) / config.channelsOut;
    const int chanlow = config.chanlow + chanwidth * rank;
    const int chanhigh = std::min(chanlow + chanwidth * (rank + 1) - 1, config.chanhigh);

    // TODO: Concatenate measurement sets
    // For now, just take the first mset
    MeasurementSet mset(
        config.msets, chanlow, chanhigh,
        std::numeric_limits<double>::min(), std::numeric_limits<double>::max()
    );

    queen.send(0, 0, mset.midfreq());

    auto uvdata = [&] () -> std::generator<UVDatum<P>> {
        for (auto& uvdatum : mset) {
            co_yield static_cast<UVDatum<P>>(uvdatum);
        }
    };

    fmt::println("Worker [{}/{}]: Calculating visibility weights...", rank + 1, hivesize);
    std::shared_ptr<Weighter<P>> weighter {};
    if (config.weight == "uniform") {
        weighter = std::make_shared<Uniform<P>>(uvdata(), config.gridconf.padded());
    } else if (config.weight == "natural") {
        weighter = std::make_shared<Natural<P>>(uvdata(), config.gridconf.padded());
    } else if (config.weight == "briggs") {
        weighter = std::make_shared<Briggs<P>>(uvdata(), config.gridconf.padded(), config.robust);
    } else {
        std::runtime_error(fmt::format("Unknown weight: {}", config.weight));
    }

    // TODO: Add max duration

    auto phaseCenter = mset.phaseCenter();
    fmt::println(
        "Phase center set to RA={:.2f} Dec={:.2f}",
        rad2deg(phaseCenter.ra), rad2deg(phaseCenter.dec)
    );

    // Construct A terms matrix for beam correction
    auto beam = Beam::getBeam<P>(mset);
    auto Aterms = beam->gridResponse(
        config.gridconf.subgrid(), phaseCenter, mset.midfreq()
    );

    // Partition data and write to disk
    fmt::println("Worker [{}/{}]: Reading and partitioning data...", rank + 1, hivesize);
    auto workunits = partition(uvdata(), config.gridconf, Aterms);

    applyWeights(*weighter, workunits);

    // Create beam power at standard grid resolution
    HostArray<StokesI<P>, 2> beamPower {config.gridconf.subgrid().shape()};
    for (size_t i {}, I = config.gridconf.subgrid().size(); i < I; ++i) {
        beamPower[i] = StokesI<P>::beamPower(Aterms[i], Aterms[i]);
    }
    beamPower = rescale(beamPower, config.gridconf.subgrid(), config.gridconf.padded());
    beamPower = resize(beamPower, config.gridconf.padded(), config.gridconf.grid());

    // Create psf
    HostArray<thrust::complex<P>, 2> psf;
    {
        mpi::Lock lock(hive);
        fmt::println(
            "Worker [{}/{}]: Constructing PSF...",
            rank + 1, hivesize
        );
        psf = invert<thrust::complex, P>(workunits, config.gridconf, true);
        queen.send(0, 0, psf);
    };
    if (hivesize > 1) save(fmt::format("psf-{:02d}.fits", rank + 1), psf);

    GridSpec gridspecPsf;
    queen.recv(0, boost::mpi::any_tag, gridspecPsf);
    psf = resize(psf, config.gridconf.grid(), gridspecPsf);

    // Initial inversion
    HostArray<StokesI<P>, 2> residual;
    {
        mpi::Lock lock(hive);
        fmt::println(
            "Worker [{}/{}]: Constructing dirty image...",
            rank + 1, hivesize
        );
        residual = invert<StokesI, P>(workunits, config.gridconf);
        queen.send(0, 0, residual);
    };
    if (hivesize > 1) save(fmt::format("dirty-{:02d}.fits", rank + 1), residual);

    // Initialize components array
    HostArray<StokesI<P>, 2> components{config.gridconf.grid().shape()};

    // This flag is set by the queen, indicating whether to proceed
    // for each major clean loop
    bool again;
    queen.recv(0, boost::mpi::any_tag, again);

    // Clean loops
    while (again) {
        // Predict
        {
            HostArray<StokesI<P>, 2> minorComponents;
            queen.recv(0, boost::mpi::any_tag, minorComponents);
            components += minorComponents;

            // Correct for beamPower
            for (size_t i {}, I = config.gridconf.grid().size(); i < I; ++i) {
                minorComponents[i] /= beamPower[i];
            }

            mpi::Lock lock(hive);

            fmt::println(
                "Worker [{}/{}]: Removing clean components from uvdata...",
                rank + 1, hivesize
            );
            predict<StokesI<P>, P>(workunits, minorComponents, config.gridconf, DegridOp::Subtract);
        }

        // Invert
        {
            mpi::Lock lock(hive);
            fmt::println(
                "Worker [{}/{}]: Constructing residual image...", rank + 1, hivesize
            );
            residual = invert<StokesI, P>(workunits, config.gridconf);
            queen.send(0, 0, residual);
        }

        // Listen for major loop termination signal from queen
        queen.recv(0, boost::mpi::any_tag, again);
    }

    queen.send(0, 0, components);

    // Write out data
    if (hivesize > 1) {
        save(fmt::format("components-{:02d}.fits", rank + 1), components);

        auto image = convolve(
            components,
            PSF<P>(psf, gridspecPsf).draw(config.gridconf.grid())
        );
        image += residual;

        save(fmt::format("image-{:02d}.fits", rank + 1), image);
    }
}

}