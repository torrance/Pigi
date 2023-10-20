#include <algorithm>
#include <generator>
#include <iterator>
#include <memory>

#include "beam.h"
#include "clean.h"
#include "config.h"
#include "channelgroup.h"
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

    // TODO: Calculate Beam for each mset
    auto Aterms = Beam::Uniform<P>().gridResponse(config.subgridspec, {0, 0}, 0);

    // Partition all msets and combine their workunits by channel
    std::vector<ChannelGroup<StokesI, P>> channelgroups;

    for (auto& [channelIndex, msetsChannelGroup] : config.msets) {
        // Concatenate all mset data for this channel group
        fmt::println(
            "Channel group [{}/{}]: Reading and partitioning data...",
            channelIndex, config.msets.size()
        );

        std::vector<WorkUnit<P>> workunitsChannelgroup;
        for (auto& mset : msetsChannelGroup) {
            auto uvdata = [&] () -> std::generator<UVDatum<P>> {
                for (auto& uvdatum : mset) {
                    co_yield static_cast<UVDatum<P>>(uvdatum);
                }
            }();

            auto workunits = partition(
                uvdata, config.gridspecPadded, config.subgridspec,
                config.padding, config.wstep, Aterms
            );

            // Append workunits to workunitsChannelGroup using move semantics
            workunitsChannelgroup.reserve(workunitsChannelgroup.size() + workunits.size());
            std::move(
                workunits.begin(), workunits.end(),
                std::back_inserter(workunitsChannelgroup)
            );
        }

        // Apply weights
        auto totalWeight = applyWeights(*weighter, workunitsChannelgroup);

        // Create psf
        fmt::println(
            "Channel group [{}/{}]: Constructing PSF...",
            channelIndex, config.msets.size()
        );
        auto psfDirtyPadded = invert<thrust::complex, P>(
            workunitsChannelgroup, config.gridspecPadded, taper, subtaper, true
        );
        auto psfDirty = resize(psfDirtyPadded, config.gridspecPadded, config.gridspec);

        // Initial inversion
        fmt::println(
            "Channel group [{}/{}]: Constructing dirty image...",
            channelIndex, config.msets.size()
        );
        auto residualPadded = invert<StokesI, P>(
            workunitsChannelgroup, config.gridspecPadded, taper, subtaper
        );
        auto residual = resize(residualPadded, config.gridspecPadded, config.gridspec);

        // Write out sub channel images
        if (channelgroups.size() > 1) {
            save(fmt::format("psf-{:02d}.fits", channelIndex), psfDirty);
            save(fmt::format("dirty-{:02d}.fits", channelIndex), residual);
        }

        // Create empty array for accumulating clean compoents across all major cycles
        HostArray<StokesI<P>, 2> components {config.gridspec.Nx, config.gridspec.Ny};

        ChannelGroup<StokesI, P> channelgroup {
            .channelIndex = channelIndex,
            .midfreq = msetsChannelGroup.front().midfreq(),
            .weights = totalWeight,
            .msets = std::move(msetsChannelGroup),
            .workunits = std::move(workunitsChannelgroup),
            .psf = std::move(psfDirty),
            .residual = std::move(residual),
            .components = std::move(components)
        };
        channelgroups.push_back(std::move(channelgroup));
    }

    // Combine dirty and psf images for full channel output
    PSF<P> psf {};
    GridSpec gridspecPsf {};
    {
        HostArray<StokesI<P>, 2> residualCombined {config.gridspec.Nx, config.gridspec.Ny};
        HostArray<thrust::complex<P>, 2> psfCombined {
            config.gridspec.Nx, config.gridspec.Ny
        };

        for (auto& channelgroup : channelgroups) {
            residualCombined += channelgroup.residual;
            psfCombined += channelgroup.psf;
        }
        residualCombined /= StokesI<P>(channelgroups.size());
        psfCombined /= thrust::complex<P>(channelgroups.size());

        save("dirty.fits", residualCombined);
        save("psf.fits", psfCombined);

        // Further crop psfs so that all pixels > 0.2 * (1 - majorgain)
        // are included in cutout. We use this smaller window to speed up PSF fitting and
        // cleaning.
        gridspecPsf = cropPsf(
            psfCombined, config.gridspec, 0.2 * (1 - config.majorgain)
        );

        for (auto& channelgroup : channelgroups) {
            channelgroup.psf = resize(channelgroup.psf, config.gridspec, gridspecPsf);
        }

        // Fit combined psf
        psfCombined = resize(psfCombined, config.gridspec, gridspecPsf);
        psf = PSF(psfCombined, gridspecPsf);
    }

    for (
        size_t imajor {}, iminor {};
        imajor < config.nMajor && iminor < config.nMinor;
        ++imajor
    ) {
        auto [minorComponents, iters, finalMajor] = clean::major<P>(
            channelgroups, config.gridspec, gridspecPsf,
            config.minorgain, config.majorgain, config.cleanThreshold,
            config.autoThreshold, config.nMinor - iminor
        );
        iminor += iters;

        for (size_t n = {}; n < channelgroups.size(); ++n) {
            auto& channelgroup = channelgroups[n];

            // Append components from this major interation
            channelgroup.components += minorComponents[n];

            fmt::println(
                "Channel group {}: Removing clean components from uvdata...", n + 1
            );
            auto minorComponentsPadded = resize(
                minorComponents[n], config.gridspec, config.gridspecPadded
            );
            predict<StokesI<P>, P>(
                channelgroup.workunits,
                minorComponentsPadded,
                config.gridspecPadded,
                taper,
                subtaper,
                DegridOp::Subtract
            );

            fmt::println(
                "Channel group {}: Constructing residual image...", n + 1
            );
            auto residualPadded = invert<StokesI, P>(
                channelgroup.workunits, config.gridspecPadded, taper, subtaper
            );
            channelgroup.residual = resize(residualPadded, config.gridspecPadded, config.gridspec);
        }

        if (finalMajor) break;
    }

    HostArray<StokesI<P>, 2> residualCombined {config.gridspec.Nx, config.gridspec.Ny};
    HostArray<StokesI<P>, 2> componentsCombined {config.gridspec.Nx, config.gridspec.Ny};

    for (size_t n {}; n < channelgroups.size(); ++n) {
        auto& channelgroup = channelgroups[n];

        residualCombined += channelgroup.residual;
        componentsCombined += channelgroup.components;

        if (channelgroups.size() > 1) {
            fmt::println("Channel group {}: Fitting psf...", n + 1);
            auto psf = PSF<P>(channelgroup.psf, gridspecPsf);

            auto image = convolve(channelgroup.components, psf.draw(config.gridspec));
            image += channelgroup.residual;

            save(fmt::format("residual-{:02d}.fits", n + 1), channelgroup.residual);
            save(fmt::format("components-{:02d}.fits", n + 1), channelgroup.residual);
            save(fmt::format("image-{:02d}.fits", n + 1), image);
        }
    }

    residualCombined /= StokesI<P>(channelgroups.size());
    componentsCombined /= StokesI<P>(channelgroups.size());

    auto imageCombined = convolve(componentsCombined, psf.draw(config.gridspec));
    imageCombined += residualCombined;

    save("residual.fits", residualCombined);
    save("components.fits", componentsCombined);
    save("image.fits", imageCombined);
}

}