#pragma once

#include <algorithm>
#include <barrier>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>

#include <boost/mpi.hpp>
#include <fmt/format.h>

#include "aterms.h"
#include "beam.h"
#include "clean.h"
#include "config.h"
#include "fits.h"
#include "gridspec.h"
#include "invert.h"
#include "logger.h"
#include "mpi.h"
#include "partition.h"
#include "predict.h"
#include "psf.h"
#include "taper.h"
#include "workunit.h"
#include "util.h"
#include "weighter.h"

namespace routines {

template <typename P>
void cleanQueen(const Config& config, boost::mpi::intercommunicator hive) {
    Logger::setName("Queen");
    Logger::debug("All shall love me and despair");

    const int hivesize = hive.remote_size();
    const auto gridconfs = config.gridconfs();

    const std::ranges::iota_view<int, int> srcs{0, hivesize};
    const std::ranges::iota_view<size_t, size_t> fieldids{0, gridconfs.size()};

    std::vector<DataTable::FreqRange> freqs(hivesize);
    for (const int src : srcs) {
        hive.recv(src, boost::mpi::any_tag, freqs[src]);
    }

    // Initialize shared values for cleaning
    // Vectors of vectors are indexed by [fieldid][channel]
    std::vector<std::vector<HostArray<thrust::complex<P>, 2>>> psfss(gridconfs.size());
    std::vector<std::vector<HostArray<StokesI<P>, 2>>> residualss(gridconfs.size());
    std::vector<clean::ComponentMap<StokesI<P>>> minorComponentsMaps;
    size_t iminor {};
    bool finalMajor {};

    // Create a barrier object that does cleaning as part of its completion function.
    // This completion function is run just by one thread.
    std::barrier cleanbarrier(fieldids.size(), [&] () -> void {
        std::vector<GridSpec> gridspecs;
        gridspecs.reserve(gridconfs.size());
        for (auto& gridconf : gridconfs) gridspecs.push_back(gridconf.grid());

        auto [_minorComponentsMaps, _iters, _finalMajor] = clean::major<P>(
            freqs, residualss, gridspecs, psfss,
            config.minorgain, config.majorgain, config.cleanThreshold,
            config.autoThreshold, config.nMinor - iminor, config.spectralparams
        );

        minorComponentsMaps = std::move(_minorComponentsMaps);
        iminor += _iters;
        finalMajor = _finalMajor;
    });

    std::vector<std::thread> threads;
    for (size_t fieldid : fieldids) {
        threads.emplace_back([&, fieldid=fieldid] {
            Logger::setName("Queen (field {}/{})", fieldid + 1, fieldids.size());
            Logger::debug("Thread created");

            auto gridconf = gridconfs[fieldid];

            // Collect beams from workers
            {
                HostArray<StokesI<P>, 2> beamPowerCombined {gridconf.grid().shape()};
                HostArray<StokesI<P>, 2> beamPower;

                for (const int src : srcs) {
                    hive.recv(src, fieldid, beamPower);
                    beamPowerCombined += beamPower;
                }
                beamPowerCombined /= StokesI<P>(hivesize);

                fits::save(
                    fmt::format("beam-field{:02d}.fits", fieldid + 1), beamPowerCombined,
                    gridconf.grid(), config.phasecenter.value()
                );
            }

            auto& psfs = psfss[fieldid];
            psfs.resize(hivesize);
            for (const int src : srcs) {
                hive.recv(src, fieldid, psfs[src]);
            }

            PSF<P> psf;
            GridSpec gridspecPsf;
            {
                // Collect psfs from workers
                HostArray<thrust::complex<P>, 2> psfCombined {gridconf.grid().shape()};
                for (const auto& psf : psfs) {
                    psfCombined += psf;
                }
                psfCombined /= thrust::complex<P>(hivesize);
                fits::save(
                    fmt::format("psf-field{:02d}.fits", fieldid + 1), psfCombined,
                    gridconf.grid(), config.phasecenter.value()
                );

                // Crop psfs to hardcoded 256 x 256 pixels. We use this smaller window to
                // speed up PSF fitting and cleaning.
                gridspecPsf = GridSpec::fromScaleLM(
                    std::min(gridconf.grid().Nx, 256ll),
                    std::min(gridconf.grid().Ny, 256ll),
                    gridconf.grid().scalel
                );

                for (auto& psf : psfs) {
                    psf = resize(psf, gridconf.grid(), gridspecPsf);
                }
                psfCombined = resize(psfCombined, gridconf.grid(), gridspecPsf);

                // Fit the combined psf to construct Gaussian PSF
                psf = PSF(psfCombined, gridspecPsf);
            }

            // Collect initial residual images from workers
            auto& residuals = residualss[fieldid];
            residuals.resize(hivesize);
            for (const int src : srcs) {
                hive.recv(src, fieldid, residuals[src]);
            }

            // Write out dirty combined
            {
                HostArray<StokesI<P>, 2> dirtyCombined {gridconf.grid().shape()};
                for (auto& residual : residuals) {
                    dirtyCombined += residual;
                }
                dirtyCombined /= StokesI<P>(hivesize);

                fits::save(
                    fmt::format("dirty-field{:02d}.fits", fieldid + 1), dirtyCombined,
                    gridconf.grid(), config.phasecenter.value()
                );
            }

            // Major clean loops
            for (
                long imajor {}, iminor {};
                imajor < config.nMajor && iminor < config.nMinor;
                ++imajor
            ) {
                // Signal continuation of clean loop to workers
                for (const int src : srcs) hive.send(src, fieldid, true);

                // Run cleaning on just one thread
                cleanbarrier.arrive_and_wait();

                // Send out clean components for subtraction from visibilities by workers
                for (const int src : srcs) {
                    hive.send(src, fieldid, minorComponentsMaps[src]);
                }

                // Gather new residuals from workers
                for (const int src : srcs) {
                    hive.recv(src, fieldid, residuals[src]);
                }

                if (finalMajor) {
                    break;
                }
            }

            // Signal end of clean loop to workers
            for (const int src : srcs) hive.send(src, fieldid, false);

            // Write out the final images to disk
            HostArray<StokesI<P>, 2> residualCombined {gridconf.grid().shape()};
            HostArray<StokesI<P>, 2> componentsCombined {gridconf.grid().shape()};

            {
                HostArray<StokesI<P>, 2> components;
                for (const int src : srcs) {
                    residualCombined += residuals[src];
                    hive.recv(src, fieldid, components);
                    componentsCombined += components;
                }
            }

            residualCombined /= StokesI<P>(hivesize);
            componentsCombined /= StokesI<P>(hivesize);

            Logger::info("Convolving final image...");
            auto imageCombined = convolve(componentsCombined, psf.draw(gridconf.grid()));
            imageCombined += residualCombined;

            Logger::info("Writing out files...");
            fits::save(
                fmt::format("residual-field{:02d}.fits", fieldid + 1), residualCombined,
                gridconf.grid(), config.phasecenter.value()
            );
            fits::save(
                fmt::format("components-field{:02d}.fits", fieldid + 1), componentsCombined,
                gridconf.grid(), config.phasecenter.value()
            );
            fits::save(
                fmt::format("image-field{:02d}.fits", fieldid + 1), imageCombined,
                gridconf.grid(), config.phasecenter.value()
            );
        });
    }

    for (auto& thread : threads) thread.join();
}

template <typename P>
void cleanWorker(
    const Config& config,
    boost::mpi::intercommunicator queen,
    boost::mpi::communicator hive
) {
    const int rank = hive.rank();
    const int hivesize = hive.size();

    Logger::setName("Worker {}/{}", rank + 1, hivesize);
    Logger::debug("Reporting for duty!");

    // Set GPU for worker
    GPU::getInstance().setID(rank % GPU::getInstance().getCount());
    Logger::debug("Selecting default GPU to ID={}", GPU::getInstance().getID());

    GPU::getInstance().setmem(static_cast<size_t>(config.gpumem * 1e9));
    Logger::debug("Maximum GPU mem = {:.1f} GB", GPU::getInstance().getmem() / 1e9);

    const auto gridconfs = config.gridconfs();

    // Calculate the channel bounds for each rank
    const int chanwidth = cld(config.chanhigh - config.chanlow, config.channelsOut);
    const int chanlow = config.chanlow + chanwidth * rank;
    const int chanhigh = std::min(chanlow + chanwidth, config.chanhigh);

    // Open msets and load data into memory
    std::vector<casacore::MeasurementSet> msets;
    for (auto& fname : config.msets) msets.push_back({fname});
    DataTable tbl(msets, {
        .chanlow=chanlow, .chanhigh=chanhigh,
        .datacolumn=config.datacolumn, .phasecenter=config.phasecenter,
    });

    queen.send(0, 0, tbl.freqrange());

    Logger::info("Calculating visibility weights...");
    std::shared_ptr<Weighter> weighter {};
    {
        // Use the padded grid of the largest field for weighting
        auto grid = std::max_element(
            gridconfs.begin(), gridconfs.end(), [] (auto& a, auto &b) {
                return a.imgNx * a.imgNy < b.imgNx * b.imgNy;
            }
        )->padded();

        if (config.weight == "uniform") {
            weighter = std::make_shared<Uniform>(tbl, grid);
        } else if (config.weight == "natural") {
            weighter = std::make_shared<Natural>(tbl, grid);
        } else if (config.weight == "briggs") {
            weighter = std::make_shared<Briggs>(tbl, grid, config.robust);
        } else {
            std::runtime_error(fmt::format("Unknown weight: {}", config.weight));
        }
    }
    applyweights(*weighter, tbl);

    std::mutex writelock;
    std::barrier barrier(gridconfs.size(), [] {});

    // The rest of the work done by workers is threaded: one thread per field
    std::vector<std::thread> threads;
    for (size_t fieldid {}; fieldid < gridconfs.size(); ++fieldid) {
        threads.emplace_back([&, fieldid=fieldid] {
            Logger::setName(
                "Worker {}/{} (field {}/{})",
                rank + 1, hivesize, fieldid + 1, gridconfs.size()
            );
            Logger::debug("Thread created");

            GPU::getInstance().resetDevice(); // needs to be reset for each new thread

            auto gridconf = gridconfs[fieldid];
            Logger::debug(
                "Using padded grid {} x {} px", gridconf.padded().Nx, gridconf.padded().Ny
            );

            auto aterms = [&] {
                // Segfaults occur without out this lock - due to Casacore issues
                // TODO: Remove this lock and add at a lower level
                std::lock_guard l(writelock);
                return mkAterms(
                    msets,
                    gridconf.subgrid(), config.maxDuration,
                    config.phasecenter.value(), tbl.midfreq()
                );
            }();

            // Partition data
            auto workunits = partition(tbl, gridconf, aterms);

            Logger::info("Constructing average beam...");
            auto beamPower = aterms.template average<StokesI, P>(tbl, workunits, gridconf);

            queen.send(0, fieldid, beamPower);
            if (hivesize > 1) fits::save(
                fmt::format("beam-field{:02d}-{:02d}.fits", fieldid + 1, rank + 1), beamPower,
                gridconf.grid(), config.phasecenter.value()
            );

            // Create psf
            auto psf = [&] {
                std::lock_guard l(writelock);
                Logger::info("Constructing PSF...");
                return invert<thrust::complex, P>(tbl, workunits, gridconf, aterms, true);
            }();
            queen.send(0, fieldid, psf);
            if (hivesize > 1) fits::save(
                fmt::format("psf-field{:02d}-{:02d}.fits", fieldid + 1, rank + 1), psf,
                gridconf.grid(), config.phasecenter.value()
            );

            // Resize psf for cleaning to hardcoded 256 x 256 pixels to be used
            // later for psf fitting
            auto gridspecPsf = GridSpec::fromScaleLM(
                std::min(gridconf.grid().Nx, 256ll),
                std::min(gridconf.grid().Ny, 256ll),
                gridconf.grid().scalel
            );
            psf = resize(psf, gridconf.grid(), gridspecPsf);

            // Initial inversion
            auto residual = [&] {
                std::lock_guard l(writelock);
                Logger::info("Constructing dirty image...");
                return invert<StokesI, P>(tbl, workunits, gridconf, aterms);
            }();
            queen.send(0, fieldid, residual);

            if (hivesize > 1) fits::save(fmt::format(
                "dirty-field{:02d}-{:02d}.fits", fieldid + 1, rank + 1
            ), residual, gridconf.grid(), config.phasecenter.value());

            // Pre-allocate the components array, used to sum components
            // from each major iteration
            HostArray<StokesI<P>, 2> components(gridconf.grid().shape());

            // This flag is set by the queen, indicating whether to proceed
            // for each major clean loop
            bool again;
            queen.recv(0, fieldid, again);

            // Clean loops
            for (int i {1}; again; ++i) {
                // Predict
                {
                    clean::ComponentMap<StokesI<P>> minorComponentsMap;
                    queen.recv(0, fieldid, minorComponentsMap);

                    std::vector<GridSpec> gridspecs;
                    for (auto& gridconf : gridconfs) gridspecs.push_back(gridconf.grid());

                    // Cleaning returns a minorComponentsMap that maps LMpx => Component
                    // We need to paint component images with these components: one image
                    // to be used in prediction, and the other as a cumulative map used
                    // in the final image construction and convolution.
                    HostArray<StokesI<P>, 2> minorComponents {gridspecs[fieldid].shape()};
                    for (auto [lmpx, val] : minorComponentsMap) {
                        auto [lpx, mpx] = lmpx;
                        auto idx = gridspecs[fieldid].LMpxToLinear(lpx, mpx);

                        // Ignore any out-of-field components
                        if (!idx) continue;

                        // Assign any valid component to the cumulative component map
                        components[*idx] += val;

                        // Assign only *our* field's components for prediction
                        if (fieldid == clean::findnearestfield(gridspecs, lmpx).value()) {
                            minorComponents[*idx] = val;
                        }
                    }

                    // Correct for beamPower
                    for (size_t j {}, J = gridconf.grid().size(); j < J; ++j) {
                        minorComponents[j] /= beamPower[j];
                    }

                    std::lock_guard l(writelock);
                    Logger::info(
                        "Removing clean components from data... (major cycle {})", i
                    );
                    predict<StokesI, P>(
                        tbl, workunits, minorComponents, gridconf,
                        aterms, DegridOp::Subtract
                    );
                }

                // Wait for all threads to finish writing to the visibilties
                // before inversion
                barrier.arrive_and_wait();

                // Invert
                residual = [&] {
                    std::lock_guard l(writelock);
                    Logger::info("Constructing residual image... (major cycle {})", i);
                    return invert<StokesI, P>(tbl, workunits, gridconf, aterms);
                }();
                queen.send(0, fieldid, residual);

                // Listen for major loop termination signal from queen
                queen.recv(0, fieldid, again);
            }

            queen.send(0, fieldid, components);

            // Write out data
            if (hivesize > 1) {
                fits::save(fmt::format(
                    "residual-field{:02d}-{:02d}.fits", fieldid + 1, rank + 1
                ), residual, gridconf.grid(), config.phasecenter.value());
                fits::save(fmt::format(
                    "components-field{:02}-{:02d}.fits", fieldid + 1, rank + 1
                ), components, gridconf.grid(), config.phasecenter.value());

                auto image = [&] {
                    std::lock_guard l(writelock);
                    return convolve(
                        components,
                        PSF<P>(psf, gridspecPsf).draw(gridconf.grid())
                    );
                }();
                image += residual;

                fits::save(fmt::format(
                    "image-field{:02}-{:02d}.fits", fieldid + 1, rank + 1
                ), image, gridconf.grid(), config.phasecenter.value());
            }

        });
    }
    for (auto& thread : threads) thread.join();
}

}