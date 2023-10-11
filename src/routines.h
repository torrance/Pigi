#include <generator>
#include <memory>

#include "beam.h"
#include "clean.h"
#include "config.h"
#include "fits.h"
#include "gridspec.h"
#include "invert.h"
#include "mset.h"
#include "predict.h"
#include "psf.h"
#include "taper.h"
#include "workunit.h"
#include "util.h"
#include "weighter.h"

namespace routines {

template <typename P>
void clean(Config& config) {
    fmt::println("Calculating visibility weights...");

    // Weighting is performed over all dat
    // So let's flatten the msets into a single uvdatum iterator
    auto uvdataAll = [&] () -> std::generator<UVDatum<P>> {
        for (auto& [_, msetsChan] : config.msets) {
            for (auto& mset : msetsChan) {
                for (auto& uvdatum : mset) {
                    co_yield static_cast<UVDatum<P>>(uvdatum);
                }
            }
        }
    }();

    std::shared_ptr<Weighter<P>> weighter {};
    if (config.weight == "uniform") {
        weighter = std::make_shared<Uniform<P>>(uvdataAll, config.gridspecPadded);
    } else if (config.weight == "natural") {
        weighter = std::make_shared<Natural<P>>(uvdataAll, config.gridspecPadded);
    } else if (config.weight == "briggs") {
        weighter = std::make_shared<Briggs<P>>(uvdataAll, config.gridspecPadded, config.robust);
    } else {
        fmt::println("Unknown weight: {}", config.weight);
        abort();
    }

    // Create grid tapers
    auto taper = kaiserbessel<P>(config.gridspecPadded);
    auto subtaper = kaiserbessel<P>(config.subgridspec);
    save("taper.fits", taper);

    auto Aterms = Beam::Uniform<P>().gridResponse(config.subgridspec, {0, 0}, 0);
    Aterms.fill({1, 0, 0, 1});

    // Partition all msets and combine their workunits
    std::vector<WorkUnit<P>> workunits;
    for (auto& [midChan, msetsChan] : config.msets) {
        for (auto& mset : msetsChan) {
            fmt::println("Parititioning data...");

            auto uvdata = [&] () -> std::generator<UVDatum<P>> {
                for (auto& uvdatum : mset) {
                    co_yield static_cast<UVDatum<P>>(uvdatum);
                }
            }();

            auto _workunits = partition(
                uvdata, config.gridspecPadded, config.subgridspec,
                config.padding, config.wstep, Aterms
            );

            applyWeights(*weighter, _workunits);

            for (auto& workunit : _workunits) {
                workunits.push_back(std::move(workunit));
            }
        }
    }

    auto psfDirtyPadded = invert<thrust::complex, P>(
        workunits, config.gridspecPadded, taper, subtaper, true
    );
    auto psfDirty = resize(psfDirtyPadded, config.gridspecPadded, config.gridspec);
    save("psf.fits", psfDirty);

    // Further crop psfDirty so that all pixels > 0.2 * (1 - majorgain)
    // are included in cutout. We use this smaller window to speed up PSF fitting and
    // cleaning
    auto [psfWindowed, gridspecPsf] = cropPsf(
        psfDirty, config.gridspec, 0.2 * (1 - config.majorgain)
    );

    PSF<P> psf(psfWindowed, gridspecPsf);
    save("psf-windowed.fits", psfWindowed);

    // Initial inversion
    auto residualPadded = invert<StokesI, P>(
        workunits, config.gridspecPadded, taper, subtaper
    );
    auto residual = resize(residualPadded, config.gridspecPadded, config.gridspec);
    save("dirty.fits", residual);

    // Create empty array for accumulating clean compoents across all major cycles
    HostArray<StokesI<P>, 2> components {config.gridspec.Nx, config.gridspec.Ny};

    for (
        size_t imajor {}, iminor {};
        imajor < config.nMajor && iminor < config.nMinor;
        ++imajor
    ) {
        auto [minorComponents, iters, finalMajor] = clean::major<P>(
            residual, config.gridspec, psfWindowed, gridspecPsf,
            config.minorgain, config.majorgain, config.cleanThreshold,
            config.autoThreshold, config.nMinor - iminor
        );

        iminor += iters;
        components += minorComponents;

        auto minorComponentsPadded = resize(
            minorComponents, config.gridspec, config.gridspecPadded
        );

        predict<StokesI<P>, P>(
            workunits,
            minorComponentsPadded,
            config.gridspecPadded,
            taper,
            subtaper,
            DegridOp::Subtract
        );

        residualPadded = invert<StokesI, P>(
            workunits, config.gridspecPadded, taper, subtaper
        );
        residual = resize(residualPadded, config.gridspecPadded, config.gridspec);

        if (finalMajor) break;
    }

    save("residual.fits", residual);
    save("components.fits", components);

    auto psfGaussian = psf.draw(config.gridspec);
    auto imgClean = convolve(components, psfGaussian);
    imgClean += residual;
    save("image.fits", imgClean);
}

}