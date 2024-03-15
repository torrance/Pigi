#pragma once

#include <limits>
#include <optional>
#include <string>
#include <vector>

// toml11 emits a lot of -Wswitch-enum warnings; temporarily suppress these
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"
#include <toml11/toml.hpp>
#pragma GCC diagnostic pop

#include "gridspec.h"
#include "logger.h"
#include "mset.h"

#define TOML11_COLORIZE_ERROR_MESSAGE = 1

namespace toml {
    template <>
    struct from<RaDec> {
        static RaDec from_toml(const value& v) {
            return {
                deg2rad(find<double>(v, "ra")), deg2rad(find<double>(v, "dec"))
            };
        }
    };

    template <>
    struct into<RaDec> {
        static toml::value into_toml(const RaDec& radec) {
            return {{"ra", rad2deg(radec.ra)}, {"dec", rad2deg(radec.dec)}};
        }
    };

    template <typename T>
    struct into<std::optional<T>> {
        static toml::value into_toml(const std::optional<T>& opt) {
            return opt.value_or(T());
        }
    };

    template <>
    struct from<Logger::Level> {
        static Logger::Level from_toml(const value& v) {
            std::string label {get<string>(v)};
            if (label == "none") return Logger::Level::none;
            if (label == "error") return Logger::Level::error;
            if (label == "warning") return Logger::Level::warning;
            if (label == "info") return Logger::Level::info;
            if (label == "debug") return Logger::Level::debug;
            if (label == "verbose") return Logger::Level::verbose;

            throw type_error(detail::format_underline(
                "unknown logging level, expected: (none|error|warning|info|debug|verbose)",
                {{v.location(), "unrecognised level"}}
            ), v.location());
        }
    };

    template <>
    struct into<Logger::Level> {
        static toml::value into_toml(const Logger::Level level) {
            switch (level) {
            case Logger::Level::none:
                return "none";
            case Logger::Level::error:
                return "error";
            case Logger::Level::warning:
                return "warning";
            case Logger::Level::info:
                return "info";
            case Logger::Level::debug:
                return "debug";
            case Logger::Level::verbose:
                return "verbose";
            }
        }
    };

    template <>
    struct from<MeasurementSet::DataColumn> {
        static MeasurementSet::DataColumn from_toml(const value& v) {
            std::string label {get<string>(v)};
            if (label == "auto") return MeasurementSet::DataColumn::automatic;
            if (label == "data") return MeasurementSet::DataColumn::data;
            if (label == "corrected") return MeasurementSet::DataColumn::corrected;
            if (label == "model") return MeasurementSet::DataColumn::model;

            throw type_error(detail::format_underline(
                "unknown datacolumn selected, expected: (auto|data|corrected|model)",
                {{v.location(), "unrecognised name"}}
            ), v.location());
        }
    };

    template <>
    struct into<MeasurementSet::DataColumn> {
        static toml::value into_toml(const MeasurementSet::DataColumn datacol) {
            switch (datacol) {
            case MeasurementSet::DataColumn::automatic:
                return "auto";
            case MeasurementSet::DataColumn::data:
                return "data";
            case MeasurementSet::DataColumn::corrected:
                return "corrected";
            case MeasurementSet::DataColumn::model:
                return "model";
            }
        }
    };
}

struct Config {
    struct Field {
        long Nx {1000};
        long Ny {1000};
        std::optional<RaDec> projectioncenter {};

        void validate() const {}

        void from_toml(const toml::value& v) {
            this->Nx = find_or(v, "width", this->Nx);
            this->Ny = find_or(v, "height", this->Ny);

            if (this->Nx < 1000) throw std::runtime_error(toml::format_error(
                "[error] Image width must be at least 1000 px",
                v.at("width"), "must be at least 1000"
            ));
            if (this->Nx % 2 != 0) throw std::runtime_error(toml::format_error(
                "[error] Image width must be evenly sized",
                v.at("width"), "must be even"
            ));

            if (this->Ny < 1000) throw std::runtime_error(toml::format_error(
                "[error] Image height must be at least 1000 px",
                v.at("height"), "must be at least 1000"
            ));
            if (this->Ny % 2 != 0) throw std::runtime_error(toml::format_error(
                "[error] Image height must be evenly sized",
                v.at("height"), "must be even"
            ));

            if (v.contains("projectioncenter")) {
                this->projectioncenter = toml::find<RaDec>(v, "projectioncenter");
            }
        }

        toml::basic_value<toml::preserve_comments> into_toml() const {
            return {
                {"width", {this->Nx, {
                    " The image size in the horizontal direction.",
                    " [1000 <= even int: pixel]",
                }}},
                {"height", {this->Ny, {
                    " The image size in the vertical direction.",
                    " [1000 <= even int: pixel]",
                }}},
                {"projectioncenter", toml::basic_value<toml::preserve_comments>(
                    this->projectioncenter, {
                    " The projection center is the associated celestial coordinate of the",
                    " the central image pixel. Typically, this will be the same as the",
                    " phase center and can be omitted (or commented out) to use the phase",
                    " center value. [float: degree]",
                })},
            };
        }
    };

    Logger::Level loglevel {Logger::Level::info};

    // Measurement set selection
    MeasurementSet::DataColumn datacolumn {MeasurementSet::DataColumn::automatic};
    int chanlow {0};
    int chanhigh {-1};
    int channelsOut {1};
    double maxDuration {0};
    std::vector<std::string> msets {};

    // Data weighting
    std::string weight {"uniform"};
    float robust {0};

    // Image
    double scale {15}; // arcseconds
    std::optional<RaDec> phasecenter {};
    std::vector<Field> fields {{.Nx = 1000, .Ny = 1000}};

    // IDG
    int precision {32};
    int kernelsize {128};
    int kernelpadding {18};
    double paddingfactor {1.5};
    int wstep {20};

    // Clean parameters
    float majorgain {0.5};
    float minorgain {0.1};
    float cleanThreshold {0};
    float autoThreshold {3.5};
    size_t nMajor {0};
    size_t nMinor {0};
    int spectralparams {2};

    void validate() {
        if (precision != 32 && precision != 64) {
            throw std::runtime_error("idg.precision must be set to 32 or 64 bits");
        }
        if (chanlow < 0) {
            throw std::runtime_error("mset.chanlow must be >= 0");
        }
        if (chanhigh <= chanlow && chanhigh != -1) {
            throw std::runtime_error("mset.chanhigh must be > mset.chanlow");
        }
        if (channelsOut < 1) {
            throw std::runtime_error("mset.channelsout must be >= 1");
        }
        if (maxDuration < 0) {
            throw std::runtime_error("beam.duration must be >= 0");
        }
        if (weight != "uniform" && weight != "natural" && weight != "briggs") {
            throw std::runtime_error("weight.scheme must be one of uniform|natural|briggs");
        }
        if (scale <= 0) {
            throw std::runtime_error("field.scale must be >= 0");
        }
        if (kernelsize < 32) {
            throw std::runtime_error("idg.kernelsize must be >= 32");
        }
        if (kernelsize % 2 != 0) {
            throw std::runtime_error("idg.kernelsize must be even");
        }
        if (kernelpadding < 0) {
            throw std::runtime_error("idg.kernelpadding must be >= 0");
        }
        if (paddingfactor < 1) {
            throw std::runtime_error("idg.paddingfactor must be >= 1");
        }
        if (wstep <= 0) {
            throw std::runtime_error("idg.wstep must be > 0");
        }
        if (!(0 < majorgain && majorgain <= 1)) {
            throw std::runtime_error("clean.majorgain must be 0 < value <= 1");
        }
        if (!(0 < minorgain && minorgain <= 1)) {
            throw std::runtime_error("clean.minorgain must be 0 < value <= 1");
        }
        if (cleanThreshold < 0) {
            throw std::runtime_error("clean.threshold must be >= 0");
        }
        if (autoThreshold < 0) {
            throw std::runtime_error("clean.autothreshold must be >= 0");
        }
        if (nMajor < 0) {
            throw std::runtime_error("clean.nmajor must be >= 0");
        }
        if (nMinor < 0) {
            throw std::runtime_error("clean.nminor must be >= 0");
        }
        if (spectralparams < 1) {
            throw std::runtime_error("clean.spectralparams must be >= 1");
        }
        if (!phasecenter) {
            // This should be set by main
            throw std::runtime_error("image.phasecenter must be set");
        }
        if (fields.empty()) {
            throw std::runtime_error("At least one [[image.fields]] must be set");
        }

        for (auto& field : fields) {
            if (!field.projectioncenter) {
                field.projectioncenter = phasecenter;
            }
            field.validate();
        }
    }

    void from_toml(const toml::value& v) {
        this->loglevel = find_or(v, "loglevel", this->loglevel);

        if (v.contains("mset")) {
            const auto tbl = toml::find(v, "mset");
            this->chanlow = find_or(tbl, "chanlow", this->chanlow);
            this->chanhigh = find_or(tbl, "chanhigh", this->chanhigh);
            this->channelsOut = find_or(tbl, "channelsout", this->channelsOut);
            this->maxDuration = find_or(tbl, "maxduration", this->maxDuration);
            this->msets = find_or(tbl, "paths", this->msets);
            this->datacolumn = find_or(tbl, "datacolumn", this->datacolumn);
        }

        if (v.contains("weight")) {
            const auto tbl = toml::find(v, "weight");
            this->weight = find_or(tbl, "scheme", this->weight);
            this->robust = find_or<double>(tbl, "robust", this->robust);
        }

        if (v.contains("idg")) {
            const auto tbl = toml::find(v, "idg");
            this->precision = find_or(tbl, "precision", this->precision);
            this->kernelpadding = find_or(tbl, "kernelpadding", this->kernelpadding);
            this->kernelsize = find_or(tbl, "kernelsize", this->kernelsize);
            this->paddingfactor = find_or(tbl, "paddingfactor", this->paddingfactor);
            this->wstep = find_or(tbl, "wstep", this->wstep);
        }

        if (v.contains("clean")) {
            const auto tbl = toml::find(v, "clean");
            this->majorgain = find_or(tbl, "majorgain", this->majorgain);
            this->minorgain = find_or(tbl, "minorgain", this->minorgain);
            this->nMajor = find_or(tbl, "nmajor", this->nMajor);
            this->nMinor = find_or(tbl, "nminor", this->nMinor);
            this->autoThreshold = find_or(tbl, "auto-threshold", this->autoThreshold);
            this->cleanThreshold = find_or(tbl, "threshold", this->cleanThreshold);
            this->spectralparams = find_or(tbl, "spectralparams", this->spectralparams);
        }

        if (v.contains("beam")) {
            const auto tbl = toml::find(v, "beam");
            this->maxDuration = find_or(tbl, "maxduration", this->maxDuration);
        }

        if (v.contains("image")) {
            const auto tbl = toml::find(v, "image");
            this->scale = find_or(tbl, "scale", this->scale);
            if (tbl.contains("phasecenter")) {
                this->phasecenter = toml::find<RaDec>(tbl, "phasecenter");
            }
            this->fields = find_or(tbl, "fields", this->fields);
        }
    }

    toml::basic_value<toml::preserve_comments> into_toml() const {
        return {
            {"idg", {
                {"kernelsize", {this->kernelsize, {
                    " The low-resolution kernel use by IDG during (de)gridding. A-terms",
                    " will be sampled at this resolution. For A-terms with complex detail,",
                    " consider increasing this size. Typical sizes range from 64 - 128",
                    " pixels. [32 <= even int: pixel]",
                }}},
                {"kernelpadding", {this->kernelpadding, {
                    " The size of the boundary around the edge of IDG kernels that is",
                    " reserved. This size is the maximum support available for gridded",
                    " visibilities. For A-terms with sharp features, consider increasing",
                    " this value. [0 <= int: pixel]",
                }}},
                {"paddingfactor", {this->paddingfactor, {
                    " Imaging and prediction will be actually be performed on fields of",
                    " size = (size * padding factor). This extra padding removes",
                    " inaccuracies that occur due to tapering by pushing the unstable",
                    " values out beyond the field of interest. [1 <= float]",
                }}},
                {"wstep", {this->wstep, {
                    " Partition visibility data by the w value into chunks that are",
                    " wstep wide. Wider values reduce the number of w-layers that must",
                    " be processed and reduce overall imaging time. However, above a",
                    " threshold, larger wstep values will result in sudden and",
                    " significant errors. [0 < int]"
                }}},
                {"precision", {this->precision, {
                    " The floating point precision used; on many GPU architectures",
                    " 32 bit is substantially faster than 64 bit. The trade-off is",
                    " reduced precision, although under most circumstances this is dwarfed",
                    " by other errors in the data.",
                    " Options: [32, 64]",
                }}},
            }},
            {"beam", {
                {"maxduration", {this->maxDuration, {
                    " The maximum duration over which the beam will be assumed to be constant.",
                    " Lower values will result in more accurate imaging, but will also fragment",
                    " the data set, resulting in longer imaging times. 0 indicates maximum",
                    " duration. [0 <= float: second]",
                }}},
            }},
            {"clean", {
                {"spectralparams", {this->spectralparams, {
                    " Fit clean peaks with clean.spectralparams Taylor terms (as a",
                    " function of frequency). A value of 1 implies a constant term (i.e.",
                    " the mean value across all channels), 2 implies a linear function,",
                    " etc., whilst clean.spectralparams >= mset.channelsout implies no",
                    " fitting. [1 >= int]",
                }}},
                {"threshold", {this->cleanThreshold, {
                    " Cleaning will terminate when the maximum (absolute) value remaining",
                    " in the residual image reaches this threshold value. [0 >= float: Jy]",
                }}},
                {"auto-threshold", {this->autoThreshold, {
                    " Cleaning will terminate when the maximum (absolute) value remaining",
                    " in the residual image reaches (auto-threshold * estimated noise).",
                    " [0 >= float: sigma]",
                }}},
                {"nminor", {this->nMinor, {
                    " The maximum number of allowed minor iterations, summed across all",
                    " major cycles. Cleaning will terminate irrspective of whether is has",
                    " met any threshold conditions. [0 <= int]",
                }}},
                {"nmajor", {this->nMajor, {
                    " The maximum number of allowed major iterations. Cleaning will",
                    " terminate irrspective of whether is has met any threshold",
                    " conditions. [0 <= int]",
                }}},
                {"minorgain", {this->minorgain, {
                    " For each minor iteratation, subtract a PSF contribution equal to ",
                    " (majorgain * detected peak value). Smaller values might (?) aid",
                    " deconvolution of complex, extended sources. [0 < float <= 1]"
                }}},
                {"majorgain", {this->majorgain, {
                    " A major cycle will continue until the residual image contains no more",
                    " values greater than ((1 - majorgain) * initial peak value).",
                    " Reasonable values range from 0.4 - 0.8. Smaller values will tend to",
                    " result in more accurate imaging, at the expense of more major clean",
                    " cycles. Larger values can be used for a well behaved PSF with small",
                    " sidelodes. [0 < float <= 1]",
                }}},
            }},
            {"weight", {
                {"scheme", {toml::string(this->weight), {
                    " The weighting scheme used for visibiltiy data.",
                    " Options: [uniform, natural, briggs]",
                }}},
                {"robust", {this->robust, {
                    " The robust factor for Briggs weighting. For other weighting schemes,",
                    " this parameter is ignored. [float]",
                }}},
            }},
            {"image", {
                {"fields", this->fields},
                {"phasecenter", toml::basic_value<toml::preserve_comments>(
                    this->phasecenter, {
                    " The phase center of this field. Visibilities will be rotated to this",
                    " new phase center; this rotation is in memory only and will not",
                    " affect input data. The phase rotation is performed as a simple 3D",
                    " matrix rotation with respect to the measurement set's stated phase",
                    " center, where both values are assumed to use the same epoch. Omit",
                    " (or comment out) this value to use the phase center of the first",
                    " measurement set. [float: degree]",
                })},
                {"scale", {this->scale, {
                    " The angular size of a pixel at the center of the field.",
                    " [0 <= float: arcsecond]",
                }}},
            }},
            {"mset", {
                {"chanhigh", {this->chanhigh, {
                    " Select channels <= chanhigh. -1 indicates maximum channel. [int]",
                }}},
                {"chanlow", {this->chanlow, {
                    " Select channels >= chanlow. [int]",
                }}},
                {"channelsout", {this->channelsOut, {
                    " Break up the channel range into channelsout equal-sized chunks.",
                    " Each segment will be independently imaged. Cleaning will perform",
                    " peak-finding across the full channel range, but the PSF will be fit",
                    " and subtracted per-channel. [1 <= int]",
                }}},
                {"datacolumn", {this->datacolumn, {
                    " The data column to be used for imaging. If set to auto, the",
                    " CORRECTED_DATA column will be imaged if it exists, otherwise the",
                    " DATA column will be used. [auto|data|corrected|model]",
                }}},
                {"paths", {toml::value(this->msets), std::vector<std::string>{
                    " Paths to the measurement sets used in imaging. These paths can also",
                    " be provided on the command line, and doing so will override any",
                    " values provided in this configuration file. [array of paths]",
                }}},
            }},
            {"loglevel", {this->loglevel, {
                " Maximum logging level. [none|error|warning|info|debug|verbose]"
            }}},
        };
    }

    std::vector<GridConfig> gridconfs() const {
        double scalelm = std::sin(deg2rad(this->scale / 3600.));

        std::vector<GridConfig> gridconfs;
        for (const auto& field : fields) {
            auto [deltal, deltam] = RaDecTolm(
                field.projectioncenter.value(), phasecenter.value()
            );

            // We snap deltal and deltam to nearest pixel value
            long long deltalpx {std::llround(deltal / scalelm)};
            long long deltampx {std::llround(deltam / scalelm)};

            deltal = deltalpx * scalelm;
            deltam = deltampx * scalelm;

            gridconfs.push_back({
                .imgNx = field.Nx, .imgNy = field.Ny,
                .imgScalelm = scalelm, .paddingfactor = paddingfactor,
                .kernelsize = kernelsize, .kernelpadding = kernelpadding,
                .wstep = static_cast<double>(wstep),
                .deltal = deltal, .deltam = deltam
            });
        }
        return gridconfs;
    }

private:
    // toml:find_or silently ignores the key:value pair if the value is the wrong type.
    // We want: if the value is present, we want it to be the correct type.
    template <typename T>
    static T find_or(const toml::value& v, const toml::key key, const T& def) {
        if (v.contains(key)) {
            return toml::find<T>(v, key);
        } else {
            return def;
        }
    }
};