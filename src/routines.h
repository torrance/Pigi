#include <algorithm>
#include <generator>
#include <iterator>
#include <memory>
#include <optional>

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
    std::shared_ptr<Weighter<P>> weighter {};

    {
        // Weighting is performed over all data
        // So let's flatten the msets into a single uvdatum iterator
        auto uvdataAll = [&] () -> std::generator<UVDatum<P>> {
            for (auto& [_, msetsChan] : config.msets) {
                for (auto& mset : msetsChan) {
                    for (auto& uvdatum : mset) {
                        co_yield static_cast<UVDatum<P>>(uvdatum);
                    }
                }
            }
        };

        if (config.weight == "uniform") {
            weighter = std::make_shared<Uniform<P>>(uvdataAll(), config.gridconf.padded());
        } else if (config.weight == "natural") {
            weighter = std::make_shared<Natural<P>>(uvdataAll(), config.gridconf.padded());
        } else if (config.weight == "briggs") {
            weighter = std::make_shared<Briggs<P>>(uvdataAll(), config.gridconf.padded(), config.robust);
        } else {
            fmt::println("Unknown weight: {}", config.weight);
            abort();
        }
    }

    // Initialize pointing; set based on mset
    std::optional<RaDec> phaseCenter;

    // Partition all msets and combine their workunits by channel
    std::vector<ChannelGroup<StokesI, P>> channelgroups;

    for (auto& [channelIndex, msetsChannelGroup] : config.msets) {
        // Concatenate all mset data for this channel group
        fmt::println(
            "Channel group [{}/{}]: Reading and partitioning data...",
            channelIndex, config.msets.size()
        );

        // Prepare accumulation variables
        LinearData<P> totalWeight {};
        HostArray<StokesI<P>, 2> avgBeam {config.gridconf.subgrid().shape()};
        std::vector<WorkUnit<P>> workunitsChannelgroup;

        for (auto& mset : msetsChannelGroup) {
            // Set pointing based on first mset
            if (!phaseCenter) {
                phaseCenter = mset.phaseCenter();
                fmt::println(
                    "Phase center set to RA={:.2f} Dec={:.2f}",
                    rad2deg(phaseCenter->ra), rad2deg(phaseCenter->dec)
                );
            }

            // TODO: Apply phase correction for any mset where pointing is offset

            // Construct A terms matrix for beam
            auto beam = Beam::getBeam<P>(mset);
            auto Aterms = beam->gridResponse(
                config.gridconf.subgrid(), *phaseCenter, mset.midfreq()
            );

            auto uvdata = [&] () -> std::generator<UVDatum<P>> {
                for (auto& uvdatum : mset) {
                    co_yield static_cast<UVDatum<P>>(uvdatum);
                }
            }();

            auto workunits = partition(uvdata, config.gridconf, Aterms);

            // Apply weights
            auto weight = applyWeights(*weighter, workunits);
            totalWeight += weight;

            // Append workunits to workunitsChannelGroup using move semantics
            workunitsChannelgroup.reserve(workunitsChannelgroup.size() + workunits.size());
            std::move(
                workunits.begin(), workunits.end(),
                std::back_inserter(workunitsChannelgroup)
            );

            // Construct Aterm power, apply weight, and add to avgBeam
            // TODO: This assumes Aterms are 2D. Handle case for non-identical beams

            for (size_t i {}, I = config.gridconf.subgrid().size(); i < I; ++i) {
                avgBeam[i] += (
                    StokesI<P>::beamPower(Aterms[i], Aterms[i]) *= static_cast<StokesI<P>>(weight)
                );
            }
        }

        // Rescale avgBeam
        avgBeam = rescale(avgBeam, config.gridconf.subgrid(), config.gridconf.padded());
        avgBeam = resize(avgBeam, config.gridconf.padded(), config.gridconf.grid());
        avgBeam /= StokesI<P>(totalWeight);

        // Create psf
        fmt::println(
            "Channel group [{}/{}]: Constructing PSF...",
            channelIndex, config.msets.size()
        );
        auto psfDirty = invert<thrust::complex, P>(
            workunitsChannelgroup, config.gridconf, true
        );

        // Initial inversion
        fmt::println(
            "Channel group [{}/{}]: Constructing dirty image...",
            channelIndex, config.msets.size()
        );
        auto residual = invert<StokesI, P>(workunitsChannelgroup, config.gridconf);

        // Write out sub channel images
        if (config.msets.size() > 1) {
            save(fmt::format("beam-{:02d}.fits", channelIndex), avgBeam);
            save(fmt::format("psf-{:02d}.fits", channelIndex), psfDirty);
            save(fmt::format("dirty-{:02d}.fits", channelIndex), residual);
        }

        // Create empty array for accumulating clean compoents across all major cycles
        HostArray<StokesI<P>, 2> components {config.gridconf.grid().shape()};

        ChannelGroup<StokesI, P> channelgroup {
            .channelIndex = channelIndex,
            .midfreq = msetsChannelGroup.front().midfreq(),
            .weights = totalWeight,
            .msets = std::move(msetsChannelGroup),
            .workunits = std::move(workunitsChannelgroup),
            .avgBeam = std::move(avgBeam),
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
        HostArray<StokesI<P>, 2> residualCombined {config.gridconf.grid().shape()};
        HostArray<thrust::complex<P>, 2> psfCombined {config.gridconf.grid().shape()};
        HostArray<StokesI<P>, 2> beamCombined {config.gridconf.grid().shape()};

        for (auto& channelgroup : channelgroups) {
            residualCombined += channelgroup.residual;
            psfCombined += channelgroup.psf;
            beamCombined += channelgroup.avgBeam;
        }
        residualCombined /= StokesI<P>(channelgroups.size());
        psfCombined /= thrust::complex<P>(channelgroups.size());
        beamCombined /= StokesI<P>(channelgroups.size());

        save("dirty.fits", residualCombined);
        save("psf.fits", psfCombined);
        save("beam.fits", beamCombined);

        // Further crop psfs so that all pixels > 0.2 * (1 - majorgain)
        // are included in cutout. We use this smaller window to speed up PSF fitting and
        // cleaning.
        gridspecPsf = cropPsf(
            psfCombined, config.gridconf.grid(), 0.2 * (1 - config.majorgain)
        );

        for (auto& channelgroup : channelgroups) {
            channelgroup.psf = resize(
                channelgroup.psf, config.gridconf.grid(), gridspecPsf
            );
        }

        // Fit combined psf
        psfCombined = resize(psfCombined, config.gridconf.grid(), gridspecPsf);
        psf = PSF(psfCombined, gridspecPsf);
    }

    for (
        size_t imajor {}, iminor {};
        imajor < config.nMajor && iminor < config.nMinor;
        ++imajor
    ) {
        auto [minorComponents, iters, finalMajor] = clean::major<P>(
            channelgroups, config.gridconf.grid(), gridspecPsf,
            config.minorgain, config.majorgain, config.cleanThreshold,
            config.autoThreshold, config.nMinor - iminor
        );
        iminor += iters;

        for (size_t n = {}; n < channelgroups.size(); ++n) {
            auto& channelgroup = channelgroups[n];

            // Create minorComponentsMap and append
            // components from this major interation
            HostArray<StokesI<P>, 2> minorComponentsMap {config.gridconf.grid().shape()};
            for (auto [idx, val] : minorComponents[n]) {
                channelgroup.components[idx] += val;
                minorComponentsMap[idx] += (val /= channelgroup.avgBeam[idx]);
            }

            fmt::println(
                "Channel group {}: Removing clean components from uvdata...", n + 1
            );
            predict<StokesI<P>, P>(
                channelgroup.workunits, minorComponentsMap,
                config.gridconf, DegridOp::Subtract
            );

            fmt::println(
                "Channel group {}: Constructing residual image...", n + 1
            );
            channelgroup.residual = invert<StokesI, P>(
                channelgroup.workunits, config.gridconf
            );

            // Save intermediate progress for debugging
            if (channelgroups.size() > 1) {
                save(fmt::format("residual-{:02d}.fits", n + 1), channelgroup.residual);
                save(fmt::format("components-{:02d}.fits", n + 1), channelgroup.components);
            } else {
                save("residual.fits", channelgroup.residual);
                save("components.fits", channelgroup.components);
            }
        }

        if (finalMajor) break;
    }

    HostArray<StokesI<P>, 2> residualCombined {config.gridconf.grid().shape()};
    HostArray<StokesI<P>, 2> componentsCombined {config.gridconf.grid().shape()};

    for (size_t n {}; n < channelgroups.size(); ++n) {
        auto& channelgroup = channelgroups[n];

        residualCombined += channelgroup.residual;
        componentsCombined += channelgroup.components;

        if (channelgroups.size() > 1) {
            fmt::println("Channel group {}: Fitting psf...", n + 1);
            auto psf = PSF<P>(channelgroup.psf, gridspecPsf);

            auto image = convolve(channelgroup.components, psf.draw(config.gridconf.grid()));
            image += channelgroup.residual;

            save(fmt::format("residual-{:02d}.fits", n + 1), channelgroup.residual);
            save(fmt::format("components-{:02d}.fits", n + 1), channelgroup.components);
            save(fmt::format("image-{:02d}.fits", n + 1), image);
        }
    }

    residualCombined /= StokesI<P>(channelgroups.size());
    componentsCombined /= StokesI<P>(channelgroups.size());

    auto imageCombined = convolve(componentsCombined, psf.draw(config.gridconf.grid()));
    imageCombined += residualCombined;

    save("residual.fits", residualCombined);
    save("components.fits", componentsCombined);
    save("image.fits", imageCombined);
}

}