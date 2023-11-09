#include <functional>
#include <limits>
#include <string>

#include <fmt/format.h>
#include <tclap/CmdLine.h>

#include "config.h"
#include "gridspec.h"
#include "mset.h"
#include "routines.h"

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
    Config config;

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
            "must be even integer > 0", "int", [](auto val) {
                return val % 2 == 0 && val > 0;
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

        LambdaConstraint<double> wstepConstraint(
            "must be > 0", "float", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<double> wstep(
            "", "wstep",
            "The separation between w-layers (Default: 25) [lambda]",
            false, 25, &wstepConstraint, cmd
        );

        TCLAP::ValueArg<int> chanlow(
            "", "chanlow",
            "Select a subset of channels from chanlow:chanhigh, inclusive. Channels start at 0. (Default: 0)",
            false, 0, "int", cmd
        );
        TCLAP::ValueArg<int> chanhigh(
            "", "chanhigh",
            "Select a subset of channels from chanlow:chanhigh, inclusive. Channels start at 0. (Default: max int)",
            false, std::numeric_limits<int>::max(), "int", cmd
        );

        LambdaConstraint<int> channelsOutConstraint(
            "must be >= 1", "int", [](auto val) {
                return val >= 1;
            }
        );
        TCLAP::ValueArg<int> channelsOut(
            "", "channelsout",
            "Split the bandwidth into N subbands and image independently and clean jointly. (Default: 1)",
            false, 1, &channelsOutConstraint, cmd
        );

        LambdaConstraint<double> maxDurationConstraint(
            "must be > 0", "float", [](auto val) {
                return val > 0;
            }
        );
        TCLAP::ValueArg<double> maxDuration(
            "", "maxduration",
            "Ensure data of no more than maxduration is grouped together. This can be used to ensure the primary beam is assumed constant for no longer than this time. (Default: max float) [second]",
            false, std::numeric_limits<double>::max(), &maxDurationConstraint, cmd
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
            "Padding applied to images during inversion and prediction. (Default: 1.5)",
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
            "", "nminor",
            "Maximum total number of minor cycles allowed, summed over all major clean cycles. -1 implies no limit. (Default: 0)",
            false, -1, &cleanIterConstraint, cmd
        );
        TCLAP::ValueArg<int> nMajor(
            "", "nmajor",
            "Maximum total number of major cycles allowed. -1 implies no limit. (Default: -1)",
            false, -1, &cleanIterConstraint, cmd
        );

        TCLAP::UnlabeledMultiArg<std::string> fnames(
            "fnames",
            "Measurement set path(s)",
            true, "path, [path...]", cmd
        );

        cmd.parse(argc, argv);

        config.precision = precision.getValue();
        config.chanlow = chanlow.getValue();
        config.chanhigh = chanhigh.getValue();
        config.channelsOut = channelsOut.getValue();
        config.maxDuration = maxDuration.getValue();
        config.gridconf = {
            .imgNx = size.getValue(), .imgNy = size.getValue(),
            .imgScalelm = deg2rad(scale.getValue() / 3600),
            .paddingfactor = padding.getValue(),
            .kernelsize = kernelsize.getValue(), .kernelpadding = kernelpadding.getValue(),
            .wstep = wstep.getValue()
        };
        config.weight = weight.getValue();
        config.robust = robust.getValue();
        config.minorgain = minorgain.getValue();
        config.majorgain = majorgain.getValue();
        config.cleanThreshold = cleanThreshold.getValue();
        config.autoThreshold = autoThreshold.getValue();
        config.nMinor = nMinor.getValue() < 0 ?
                        std::numeric_limits<size_t>::max() : nMinor.getValue();
        config.nMajor = nMajor.getValue() < 0 ?
                        std::numeric_limits<size_t>::max() : nMajor.getValue();

        config.msets = MeasurementSet::partition(
            fnames.getValue(), chanlow.getValue(), chanhigh.getValue(),
            config.channelsOut, config.maxDuration
        );

    } catch (TCLAP::ArgException &e) {
        fmt::println("Error: {} for argument {}", e.error(), e.argId());
    }

    if (config.precision == 32) {
        routines::clean<float>(config);
    } else {
        routines::clean<double>(config);
    }
}