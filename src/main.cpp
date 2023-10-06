#include <functional>
#include <memory>
#include <string>

#include <fmt/format.h>
#include <tclap/CmdLine.h>

#include "beam.h"
#include "clean.h"
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


class Config {
public:
    int precision;
    GridSpec gridspec;
    GridSpec gridspecPadded;
    GridSpec subgridspec;
    int wstep;
    int kernelpadding;
    MeasurementSet mset;
    std::string weight;
    float robust;
    float padding;

    // Clean parameters
    float majorgain;
    float minorgain;
    float cleanThreshold;
    float autoThreshold;
    size_t nMajor;
    size_t nMinor;

    static Config& get() {
        static Config config;
        return config;
    }

    // Singleton pattern: remove copy constructor and copy assignment
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

private:
    // Singleton patter: private constructor
    Config() {}
};

template <typename P>
void cleanroutine(Config& config) {
    auto uvdata = lazytransform(config.mset, [] (const UVDatum<double>& uvdatum) {
        return static_cast<UVDatum<P>>(uvdatum);
    });

    fmt::println("Calculating visibility weights...");
    std::shared_ptr<Weighter<P>> weighter {};
    if (config.weight == "uniform") {
        weighter = std::make_shared<Uniform<P>>(uvdata, config.gridspecPadded);
    } else if (config.weight == "natural") {
        weighter = std::make_shared<Natural<P>>(uvdata, config.gridspecPadded);
    } else if (config.weight == "briggs") {
        weighter = std::make_shared<Briggs<P>>(uvdata, config.gridspecPadded, config.robust);
    } else {
        fmt::println("Unknown weight: {}", config.weight);
        abort();
    }

    // auto Aterms = Beam::Uniform<P>().gridResponse(config.subgridspec, {0, 0}, 0);
    HostArray<ComplexLinearData<P>, 2> Aterms {config.subgridspec.Nx, config.subgridspec.Ny};
    Aterms.fill({1, 0, 0, 1});

    fmt::println("Parititioning data...");
    auto workunits = partition(
        uvdata, config.gridspecPadded, config.subgridspec,
        config.padding, config.wstep, Aterms
    );

    applyWeights(*weighter, workunits);

    auto taper = kaiserbessel<P>(config.gridspecPadded);
    auto subtaper = kaiserbessel<P>(config.subgridspec);

    save("taper.fits", taper);

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
            {
                .minorgain = config.minorgain,
                .majorgain = config.majorgain,
                .threshold = config.cleanThreshold,
                .autothreshold = config.autoThreshold,
                .niter = config.nMinor - iminor
            }
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

template <typename T>
class LambdaConstraint : public TCLAP::Constraint<T> {
public:
    LambdaConstraint(std::string desc, std::string id, std::function<bool(T)> fn) :
        desc(desc), id(id), fn(fn) {}

    std::string description() const override { return desc; }
    std::string shortID() const override { return id; }
    bool check(const T& val) const override { return fn(val); }

private:
    std::string desc;
    std::string id;
    std::function<bool(T)> fn;
};


int main(int argc, char** argv) {
    try {
        TCLAP::CmdLine cmd("Pigi: the Parallel Interferometric GPU Imager", ' ', "dev");

        LambdaConstraint<int> sizeConstraint(
            "must be even integer >= 1000", "int", [](auto val) {
                return val >= 1000 && val % 2 == 0;
            }
        );
        TCLAP::ValueArg<int> size(
            "", "size",
            "Image size [px]",
            true, 1000, &sizeConstraint, cmd
        );

        LambdaConstraint<float> scaleConstraint(
            "must be > 0", "float", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<float> scale(
            "", "scale",
            "Image scale [arcsec]",
            true, 1000, &scaleConstraint, cmd
        );

        LambdaConstraint<int> kernelsizeConstraint(
            "must be > 0", "int", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<int> kernelsize(
            "", "kernelsize",
            "The size of the subgrid used during gridding (default: 96) [px]",
            false, 96, &kernelsizeConstraint, cmd
        );

        LambdaConstraint<int> kernelpaddingConstraint(
            "must be > 0", "int", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<int> kernelpadding(
            "", "kernelpadding",
            "The padding around the subgrid edge that is reserved (default: 18) [px]",
            false, 18, &kernelpaddingConstraint, cmd
        );

        LambdaConstraint<int> wstepConstraint(
            "must be > 0", "int", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<int> wstep(
            "", "wstep",
            "The separation between w-layers (Default: 25) [lambda]",
            false, 25, &wstepConstraint, cmd
        );

        TCLAP::ValueArg<int> chanlow(
            "", "chanlow",
            "Select a subset of channels from chanlow:chanhigh, inclusive. Channels start at 0. (Default: all channels)",
            false, -1, "int", cmd
        );
        TCLAP::ValueArg<int> chanhigh(
            "", "chanhigh",
            "Select a subset of channels from chanlow:chanhigh, inclusive. Channels start at 0. (Default: all channels)",
            false, -1, "int", cmd
        );

        std::vector<std::string> weightValues {"natural", "uniform", "briggs"};
        TCLAP::ValuesConstraint<std::string> weightConstraint(weightValues);
        TCLAP::ValueArg<std::string> weight(
            "", "weight",
            "Visbility weighting scheme. (Default: uniform)",
            false, "uniform", &weightConstraint, cmd
        );

        TCLAP::ValueArg<float> robust(
            "", "robust",
            "Brigg's robust weighting (Default: 0)",
            false, 0, "float", cmd
        );

        std::vector<int> precisionValues {32, 64};
        TCLAP::ValuesConstraint<int> precisionConstraint(precisionValues);
        TCLAP::ValueArg<int> precision(
            "", "precision",
            "Floating point precision. (Default: 32 bit)",
            false, 32, &precisionConstraint, cmd
        );

        LambdaConstraint<float> paddingConstraint(
            "must be >= 1", "float", [] (auto val) {
                return val >= 1;
            }
        );
        TCLAP::ValueArg<float> padding(
            "", "padding",
            "Padding applied to images during inversion and predicition. (Default: 1.5)",
            false, 1.5, &paddingConstraint, cmd
        );

        LambdaConstraint<float> gainConstraint(
            "must lie in range (0, 1)", "float", [] (auto val) {
                return 0 <= val && val <= 1;
            }
        );
        TCLAP::ValueArg<float> majorgain(
            "", "major-gain",
            "The maximum gain subtracted from the dirty image during each major cleaning cycle. (Default: 0.8)",
            false, 0.8, &gainConstraint, cmd
        );
        TCLAP::ValueArg<float> minorgain(
            "", "minor-gain",
            "The gain applied to PSF subtraction for each minor clean cycle. (Default: 0.1)",
            false, 0.1, &gainConstraint, cmd
        );

        LambdaConstraint<float> thresholdConstraint(
            "must be >= 0", "float", [] (auto val) {
                return val >= 0;
            }
        );
        TCLAP::ValueArg<float> cleanThreshold(
            "", "clean-threshold",
            "Cleaning will terminate when the peak flux in the residual image reaches this (absolute) threshold. (Default: 0) [Jy]",
            false, 0,  &thresholdConstraint, cmd
        );
        TCLAP::ValueArg<float> autoThreshold(
            "", "auto-threshold",
            "Cleaning will terminate when the peak flux in the residual image reaches this factor of estimated image noise. (Default: 3) [sigma]",
            false, 3, &thresholdConstraint, cmd
        );

        LambdaConstraint<int> cleanIterConstraint(
            "must be >= 0", "int", [] (auto val) {
                return val >= 0;
            }
        );
        TCLAP::ValueArg<int> nMinor(
            "", "Nminor",
            "Maximum total number of minor cycles allowed, summed over all major clean cycles. -1 implies no limit. (Default: 0)",
            false, -1, &cleanIterConstraint, cmd
        );
        TCLAP::ValueArg<int> nMajor(
            "", "Nmajor",
            "Maximum total number of major cycles allowed. -1 implies no limit. (Default: -1)",
            false, -1, &cleanIterConstraint, cmd
        );

        TCLAP::UnlabeledMultiArg<std::string> fnames(
            "fnames",
            "Measurement set path(s)",
            true, "path, [path...]", cmd
        );

        cmd.parse(argc, argv);

        auto& config = Config::get();

        int sizePadded = size.getValue() * padding.getValue();

        auto gridspec = GridSpec::fromScaleLM(
            size.getValue(), size.getValue(), deg2rad(scale.getValue() / 3600)
        );
        auto gridspecPadded = GridSpec::fromScaleLM(
            sizePadded, sizePadded, deg2rad(scale.getValue() / 3600)
        );
        auto subgridspec = GridSpec::fromScaleUV(
            kernelsize.getValue(), kernelsize.getValue(), gridspecPadded.scaleuv
        );

        config.precision = precision.getValue();
        config.gridspec = gridspec;
        config.gridspecPadded = gridspecPadded;
        config.subgridspec = subgridspec;
        config.wstep = wstep.getValue();
        config.kernelpadding = kernelpadding.getValue();
        config.weight = weight.getValue();
        config.robust = robust.getValue();
        config.padding = padding.getValue();
        config.minorgain = minorgain.getValue();
        config.majorgain = majorgain.getValue();
        config.cleanThreshold = cleanThreshold.getValue();
        config.autoThreshold = autoThreshold.getValue();
        config.nMinor = nMinor.getValue() < 0 ?
                        std::numeric_limits<size_t>::max() : nMinor.getValue();
        config.nMajor = nMajor.getValue() < 0 ?
                        std::numeric_limits<size_t>::max() : nMajor.getValue();

        config.mset = MeasurementSet(
            fnames.getValue().front(),
            MeasurementSet::Config{
                .chanlow = chanlow.getValue(),
                .chanhigh = chanhigh.getValue()
            }
        );

    } catch (TCLAP::ArgException &e) {
        fmt::println("Error: {} for argument {}", e.error(), e.argId());
    }

    auto& config = Config::get();

    if (config.precision == 32) {
        cleanroutine<float>(config);
    } else {
        cleanroutine<double>(config);
    }
}