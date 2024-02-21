#pragma once

#include <limits>
#include <string>
#include <vector>

// toml11 emits a lot of -Wswitch-enum warnings; temporarily suppress these
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"
#include <toml11/toml.hpp>
#pragma GCC diagnostic pop

#include "gridspec.h"
#include "mset.h"

#define TOML11_COLORIZE_ERROR_MESSAGE = 1

struct Config {
    int precision {32};

    // Measurement set selection
    int chanlow {0};
    int chanhigh {-1};
    int channelsOut {1};
    double maxDuration {0};
    std::vector<std::string> msets;

    // Data weighting
    std::string weight {"uniform"};
    float robust {0};

    // Image
    int size {1000};
    double scale {15}; // arcseconds
    RaDec phasecenter {};
    bool phaserotate {false};
    RaDec projectioncenter {
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()
    };

    // IDG
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
        if (size < 1000) {
            throw std::runtime_error("field.size must be >= 1000");
        }
        if (scale <= 0) {
            throw std::runtime_error("field.scale must be >= 0");
        }
        if (kernelsize < 32) {
            throw std::runtime_error("idg.kernelsize must be >= 32");
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
    }

    void from_toml(const toml::value& v) {
        this->precision = find_or(v, "precision", this->precision);

        if (v.contains("mset")) {
            const auto tbl = toml::find(v, "mset");
            this->chanlow = find_or(tbl, "chanlow", this->chanlow);
            this->chanhigh = find_or(tbl, "chanhigh", this->chanhigh);
            this->channelsOut = find_or(tbl, "channelsout", this->channelsOut);
            this->maxDuration = find_or(tbl, "maxduration", this->maxDuration);
            this->msets = find_or(tbl, "paths", this->msets);
        }

        if (v.contains("weight")) {
            const auto tbl = toml::find(v, "weight");
            this->weight = find_or(tbl, "scheme", this->weight);
            this->robust = find_or<double>(tbl, "robust", this->robust);
        }

        if (v.contains("idg")) {
            const auto tbl = toml::find(v, "idg");
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
        }

        if (v.contains("beam")) {
            const auto tbl = toml::find(v, "beam");
            this->maxDuration = find_or(tbl, "maxduration", this->maxDuration);
        }

        if (v.contains("field")) {
            const auto tbl = toml::find(v, "field");
            this->size = find_or(tbl, "size", this->size);
            this->scale = find_or(tbl, "scale", this->scale);
            this->phaserotate = find_or(tbl, "phaserotate", this->phaserotate);

            if (tbl.contains("phasecenter")) {
                const auto subtbl = toml::find(tbl, "phasecenter");
                this->phasecenter.ra = deg2rad(
                    find_or(subtbl, "ra", rad2deg(this->phasecenter.ra))
                );
                this->phasecenter.dec = deg2rad(
                    find_or(subtbl, "dec", rad2deg(this->phasecenter.dec))
                );
            }

            if (tbl.contains("projectioncenter")) {
                const auto subtbl = toml::find(tbl, "projectioncenter");
                this->projectioncenter.ra = deg2rad(
                    find_or(subtbl, "ra", rad2deg(this->projectioncenter.ra))
                );
                this->projectioncenter.dec = deg2rad(
                    find_or(subtbl, "dec", rad2deg(this->projectioncenter.dec))
                );
            }
        }
    }

    toml::basic_value<toml::preserve_comments> into_toml() const {
        return {
            {"idg", {
                {"kernelsize", {this->kernelsize, {
                    " The low-resolution kernel use by IDG during (de)gridding. A-terms",
                    " will be sampled at this resolution. For A-terms with complex detail,",
                    " consider increasing this size. Typical sizes range from 64 - 128",
                    " pixels. [32 <= int: pixel]",
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
            {"field", {
                {"size", {this->size, {
                    " The image size (size x size). [1000 <= int: pixel]",
                }}},
                {"scale", {this->scale, {
                    " The angular size of a pixel at the center of the field.",
                    " [0 <= float: arcsecond]",
                }}},
                {"projectioncenter", toml::basic_value<toml::preserve_comments>({
                    {"dec", rad2deg(this->projectioncenter.dec)},
                    {"ra", rad2deg(this->projectioncenter.ra)}
                }, {
                    " The projection center is the associated celestial coordinate of the",
                    " the central image pixel. Typically, this will be the same as the",
                    " phase center and can be omitted use the phase center value.",
                    " [float: degree]",
                })},
                {"phasecenter", toml::basic_value<toml::preserve_comments>({
                    {"dec", rad2deg(this->phasecenter.dec)},
                    {"ra", rad2deg(this->phasecenter.ra)}
                }, {
                    " The phase center of this field. Visibilities will be rotated to this",
                    " new phase center in memory; this will not affect input data. The",
                    " phase rotation is performed as a simple 3D matrix rotation with",
                    " respect to the mseaurement set's stated phase center and this value,",
                    " where both are assumed to use the same epoch. [float: degree]",
                })},
                {"phaserotate", {this->phaserotate, {
                    " Enables phase rotation of visibilities to new field.phasecenter. If",
                    " disabled, all input data must be phased to the same phase center and",
                    " the first phase center will be assumed valid for all data. [bool]",
                }}},
            }},
            {"mset", {
                {"chanhigh", {this->chanhigh, {
                    " Select channels <= chanhigh. [int]",
                }}},
                {"chanlow", {this->chanlow, {
                    " Select channels >= chanlow. -1 indicates maximum channel. [int]",
                }}},
                {"channelsout", {this->channelsOut, {
                    " Break up the channel range into channelsout equal-sized chunks.",
                    " Each segment will be independently imaged. Cleaning will perform",
                    " peak-finding across the full channel range, but the PSF will be fit",
                    " and subtracted per-channel. [1 <= int]",
                }}},
                {"paths", {toml::value(this->msets), std::vector<std::string>{
                    " Paths to the measurement sets used in imaging. These paths can also",
                    " be provided on the command line, and doing so will override any",
                    " values provided in this configuration file. [array of paths]",
                }}},
            }},
        };
    }

    GridConfig gridconf() const {
        double scalelm = std::asin(deg2rad(this->scale / 3600.));
        auto [deltal, deltam] = RaDecTolm(this->projectioncenter, this->phasecenter);

        return {
            .imgNx = this->size, .imgNy = this->size, .imgScalelm = scalelm,
            .paddingfactor = this->paddingfactor, .kernelsize = this->kernelsize,
            .kernelpadding = this->kernelpadding, .wstep = static_cast<double>(this->wstep),
            .deltal = deltal, .deltam = deltam
        };
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