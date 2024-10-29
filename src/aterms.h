#pragma once

#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/functional/hash.hpp>
#include <fitsio.h>
#include <thrust/complex.h>

#include "beam.h"
#include "logger.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"
#include "workunit.h"

namespace Aterms {

template <typename S>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
class Interface {
public:
    using aterm_t = std::shared_ptr<HostArray<ComplexLinearData<S>, 2>>;

    virtual std::tuple<Interval, aterm_t> get(const double mjd, const int antid);

    template <template<typename> typename T>
    HostArray<T<S>, 2> average(
        DataTable& tbl, std::vector<WorkUnit>& workunits, GridConfig gridconf
    ) {
        // All intermediate weights and beams are calculated at full double precision
        // to avoid FP errors with large weight sums.

        auto subgridspec = gridconf.subgrid();
        auto gridspec = gridconf.grid();
        auto paddedspec = gridconf.padded();

        // Accumulate data weights per Aterm
        using Key = std::pair<aterm_t, aterm_t>;
        struct KeyHasher {
            std::size_t operator()(const Key& key) const {
                size_t seed {};
                boost::hash_combine(seed, std::get<0>(key));
                boost::hash_combine(seed, std::get<1>(key));
                return seed;
            }
        };
        std::unordered_map<Key, T<double>, KeyHasher> atermweights;

        for (auto& w : workunits) {
            auto [ant1, ant2] = w.baseline;
            auto [intervalleft, aleft] = this->get(w.time, ant1);
            auto [intervalright, aright] = this->get(w.time, ant2);

            // Add weight contribution for this pair of beams
            T<double>& weight = atermweights[{aleft, aright}];
            for (size_t irow {w.rowstart}; irow < w.rowend; ++irow) {
                for (size_t ichan {w.chanstart}; ichan < w.chanend; ++ichan) {
                    weight += static_cast<T<double>>(tbl.weights(irow, ichan));
                }
            }
        }

        // Calculate weight to normalize average beam
        T<double> totalWeight {};
        for (auto [atermpair, weight] : atermweights) totalWeight += weight;

        // Now for each baseline pair, compute the power and append to total beam power
        HostArray<T<double>, 2> beamPower64 {subgridspec.shape()};
        for (auto [atermpair, weight] : atermweights) {
            auto [aleft, aright] = atermpair;
            weight /= totalWeight;

            for (size_t i {}, I = subgridspec.size(); i < I; ++i) {
                beamPower64[i] += T<double>::beamPower(
                    static_cast<ComplexLinearData<double>>((*aleft)[i]),
                    static_cast<ComplexLinearData<double>>((*aright)[i])
                ) *= weight;
            }
        }

        // Now rescale and resize
        auto beamPower = static_cast<HostArray<T<S>, 2>>(beamPower64);
        beamPower = rescale(beamPower, subgridspec, paddedspec);
        beamPower = resize(beamPower, paddedspec, gridspec);
        return beamPower;
    }
};

template <typename S>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
class StaticCorrections : public Aterms::Interface<S> {
public:
    using typename Interface<S>::aterm_t;

    StaticCorrections(const HostSpan<ComplexLinearData<S>, 2> aterms) {
        this->aterms = std::make_shared<HostArray<ComplexLinearData<S>, 2>>(aterms);
    }

    std::tuple<Interval, aterm_t> get(const double, const int) override {
        return {
            Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            },
            aterms
        };
    }

private:
    aterm_t aterms;
};

template <typename S>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
class BeamCorrections : public Aterms::Interface<S> {
public:
    using typename Interface<S>::aterm_t;

    BeamCorrections(
        const std::vector<casacore::MeasurementSet>& msets,
        const GridSpec& gridspec,
        double maxDuration,
        const RaDec& gridorigin,
        const double freq
    ) {
        if (maxDuration <= 0) maxDuration = std::numeric_limits<double>::infinity();

        // For now, require all msets be the same telescope
        std::string telescope;
        for (size_t i {}; auto& mset : msets) {
            std::string tname = casacore::MSObservationColumns(
                mset.observation()
            ).telescopeName().get(0);

            if (i == 0) telescope = tname;
            if (telescope != tname) throw std::runtime_error(fmt::format(
                "Require all msets to be same telescope. Got {}; expected {}", tname, telescope
            ));
        }

        // TODO: Remove these explicit branches on telescope type
        if (telescope == "MWA") {
            // Get MWA delays
            using mwadelay_t = std::tuple<double, double, std::array<uint32_t, 16>>;
            std::vector<mwadelay_t> delays;
            for (auto& mset : msets) {
                auto mwaTilePointingTbl = mset.keywordSet().asTable("MWA_TILE_POINTING");
                if (mwaTilePointingTbl.nrow() != 1) std::runtime_error(fmt::format(
                    "Found {} MWA delay rows in {}; expected 1",
                    mwaTilePointingTbl.nrow(), mwaTilePointingTbl.tableName()
                ));

                auto intervalsCol = casacore::ArrayColumn<double>(mwaTilePointingTbl, "INTERVAL");
                auto delaysCol = casacore::ArrayColumn<int>(mwaTilePointingTbl, "DELAYS");

                auto intervalRow = intervalsCol.get(0).tovector();
                auto delaysRow = delaysCol.get(0);

                // Copy casacore array to fixed array
                std::array<uint32_t, 16> delays_uint32;
                std::copy(delaysRow.begin(), delaysRow.end(), delays_uint32.begin());

                delays.push_back(std::make_tuple(
                    intervalRow[0] / 86400., intervalRow[1] / 86400., delays_uint32
                ));
            }

            for (auto& [start, end, delays] : delays) {
                for (double t0 {start}; t0 < end; t0 += maxDuration) {
                    double t1 = std::min(t0 + maxDuration, end);

                    Interval interval(t0, t1);

                    Logger::debug(
                        "Generating MWA beam for time range {}-{} at {} MHz",
                        t0, t1, freq
                    );

                    auto jones = Beam::MWA<S>(interval.mid(), delays).gridResponse(
                        gridspec, gridorigin, freq
                    );

                    intervals.push_back(interval);
                    data.push_back(std::make_shared<HostArray<ComplexLinearData<S>, 2>>(
                        jones
                    ));
                }
            }
        } else {
            Logger::warning("Unkown telescope: defaulting to uniform beam");
            auto jones = Beam::Uniform<S>().gridResponse(gridspec, gridorigin, freq);
            intervals.push_back(Interval{
                -std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
            });
            data.push_back(std::make_shared<HostArray<ComplexLinearData<S>, 2>>(jones));
        }
    }

    std::tuple<Interval, aterm_t> get(const double mjd, const int) override {
        // For now, assume beam is invariant across stations
        for (size_t i {}; i < intervals.size(); ++i) {
            auto interval = intervals[i];
            if (interval.contains(mjd)) {
                return {interval, data[i]};
            }
        }
        throw std::runtime_error(fmt::format("No BeamCorrection found for time = {}", mjd));
    }

private:
    std::vector<Interval> intervals;
    std::vector<aterm_t> data;
};

template <typename S>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
class PhaseCorrections : public Aterms::Interface<S> {
public:
    using typename Interface<S>::aterm_t;

    PhaseCorrections(const std::vector<std::string>& paths) {
        if (paths.empty()) return;

        // We expect each PhaseCorrection fits file to have 4 dimensions:
        // [time x antenna x Nx x Ny]

        // Open each file
        std::vector<fitsfile*> fptrs;
        std::vector<std::array<long, 4>> naxess;

        for (auto& path : paths) {
            int status {};
            fitsfile* fptr {};
            fits_open_image(&fptr, path.c_str(), READONLY, &status);

            // Get file metadata
            int bitpix, naxis;
            std::array<long, 4> naxes;
            fits_get_img_param(fptr, 4,  &bitpix, &naxis, naxes.data(), &status);

            // Reorder naxes column major -> row major
            std::swap(naxes[0], naxes[3]);
            std::swap(naxes[1], naxes[2]);

            if (status !=  0) {
                char fitsmsg[FLEN_STATUS] {};
                fits_get_errstatus(status, fitsmsg);

                auto msg = fmt::format("An error occurred opening {}: {}", path, fitsmsg);
                throw std::runtime_error(msg);
            }

            // Rrequire FITS data has 4 dimensions
            if (naxis != 4) throw std::runtime_error(fmt::format(
                "Opening {} failed: expected phasecorrection file with 4 dimensions, got {}",
                path, naxis
            ));

            Logger::verbose(
                "Loaded phase correction file {} with dimensions {}x{}x{}x{}",
                path, naxes[0], naxes[1], naxes[2], naxes[3]
            );

            // Ensure all axes match with the exception of the first
            if (!naxess.empty() && !(
                naxes[1] == naxess.front()[1] &&
                naxes[2] == naxess.front()[2] &&
                naxes[3] == naxess.front()[3]
            )) throw std::runtime_error(fmt::format(
                "Phase correction FITS axes are inconsistently sized"
            ));

            if (bitpix != -32 && bitpix != -64) throw std::runtime_error(fmt::format(
                "Opening {} failed: expected image with single or double precision data,"
                " got {}-bit integer data instead", path, bitpix
            ));

            // Validate outer axis is: time [s]
            char value[FLEN_CARD] {};
            fits_read_key(fptr, TSTRING, "CTYPE1", value, nullptr, &status);
            if (std::string_view(value) != "TIME") throw std::runtime_error(fmt::format(
                "Opening {} failed: expected CTYPE1=TIME, got '{}'", path, value
            ));
            fits_read_key(fptr, TSTRING, "CUNIT1", value, nullptr, &status);
            if (std::string_view(value) != "s") throw std::runtime_error(fmt::format(
                "Opening {} failed: expected CUNIT1=s, got '{}'", path, value
            ));

            // Get intervals from headers
            int crpix1;
            fits_read_key(fptr, TINT, "CRPIX1", &crpix1, nullptr, &status);
            double crval1;
            fits_read_key(fptr, TDOUBLE, "CRVAL1", &crval1, nullptr, &status);
            double cdelt1;
            fits_read_key(fptr, TDOUBLE, "CDELT1", &cdelt1, nullptr, &status);

            if (status !=  0) {
                char fitsmsg[FLEN_STATUS] {};
                fits_get_errstatus(status, fitsmsg);

                auto msg = fmt::format("An error occurred opening {}: {}", path, fitsmsg);
                throw std::runtime_error(msg);
            }

            for (long i {}; i < naxes[0]; ++i) {
                // crpix1 is 1 indexed!
                intervals.push_back(Interval {
                    crval1 + (i + 1 - crpix1) * cdelt1, crval1 + (i + 2 - crpix1) * cdelt1
                });
            }

            fptrs.push_back(fptr);
            naxess.push_back(naxes);
        }

        // Create data array with axes that match all FITS files
        std::array<long long, 4> dims;
        for (size_t i {}; i < naxess.front().size(); ++i) dims[i] = naxess.front()[i];
        dims[0] = intervals.size();
        data = HostArray<S, 4>(dims);

        // Read data into array for each file
        for (size_t i {}, offset {}; i < fptrs.size(); ++i) {
            fitsfile* fptr = fptrs[i];
            int status {};

            int datatype = std::is_same_v<S, float> ? TFLOAT : TDOUBLE;
            std::array<long, 4> fpixel {1, 1, 1, 1};  // 1-indexed
            long nelements = 1;
            for (long n : naxess[i]) nelements *= n;
            S nulval {};
            int anynul;

            fits_read_pix(
                fptr, datatype, fpixel.data(), nelements, &nulval,
                data.data() + offset, &anynul, &status
            );

            offset += nelements;

            if (nulval != 0) throw std::runtime_error(fmt::format(
                "Opening {} failed: NULL values detected in data", paths[i]
            ));

            fits_close_file(fptr, &status);

            if (status != 0) {
                char fitsmsg[FLEN_STATUS] {};
                fits_get_errstatus(status, fitsmsg);

                auto msg = fmt::format("An error occurred opening {}: {}", paths[i], fitsmsg);
                throw std::runtime_error(msg);
            }
        }
    }

    std::tuple<Interval, aterm_t> get(const double mjd, const int antid) override {
        for (size_t i {}; i < intervals.size(); ++i) {
            if (intervals[i].contains(mjd)) {
                HostSpan<S, 2> phases = data(i)(antid);
                auto phasecorrections = std::make_shared<HostArray<ComplexLinearData<S>, 2>>(
                    phases.shape()
                );

                for (size_t j {}; j < phases.size(); ++j) {
                    thrust::complex<S> phasor {
                        std::cos(phases[j]), std::sin(phases[j])
                    };

                    (*phasecorrections)[j].xx = phasor;
                    (*phasecorrections)[j].yy = phasor;
                }

                return {intervals[i], phasecorrections};
            }
        }
        throw std::runtime_error(fmt::format("No PhaseCorrection found for time = {}", mjd));
    }

private:
    std::vector<Interval> intervals;
    HostArray<S, 4> data;  // [time x ants x Nx x Ny]
};

template <typename S>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
class CombinedCorrections : public Aterms::Interface<S> {
public:
    using typename Interface<S>::aterm_t;

    CombinedCorrections(
        const std::vector<casacore::MeasurementSet>& msets,
        const GridSpec& gridspec,
        double maxDuration,
        const RaDec& gridorigin,
        const double freq,
        const std::vector<std::string>& phasecorrectionpaths
    ) : beamcorrections(msets, gridspec, maxDuration, gridorigin, freq) {
        if (!phasecorrectionpaths.empty()) {
            phasecorrections = PhaseCorrections<S>(phasecorrectionpaths);
        }
    }

    std::tuple<Interval, aterm_t> get(const double mjd, const int antid) override {
        // First, check existing cache of beams
        for (auto& [interval, aterm] : aterms[antid]) {
            if (interval.contains(mjd)) return {interval, aterm};
        }

        // Otherwise, we need to construct a beam for this antid, interval combination
        // These functions will throw exceptions if no correction exists for the given
        // time and antenna
        auto [interval, beamcorrection] = beamcorrections.get(mjd, antid);

        // Create shared pointer to beam correction term
        auto combined = std::make_shared<HostArray<ComplexLinearData<S>, 2>>(*beamcorrection);

        // Optionally conjoin phase correction values
        if (phasecorrections) {
            auto [phaseinterval, phasecorrection] = phasecorrections->get(mjd, antid);
            if (combined->shape() != phasecorrection->shape()) throw std::runtime_error(
                fmt::format(
                    "Beam correction array ({} x {}) and phase correction array ({} x {}) dimensions do not match",
                    combined->size(0), combined->size(1),
                    phasecorrection->size(0), phasecorrection->size(1)
                )
            );

            for (size_t i {}; i < combined->size(); ++i) {
                (*combined)[i] = matmul((*combined)[i], (*phasecorrection)[i]);
            }

            // Create a new interval that is the intersection beam and
            // phasecorrection intervals
            interval = {
                std::max(interval.start, phaseinterval.start),
                std::min(interval.end, phaseinterval.end)
            };
        }

        // Cache result
        aterms[antid].push_back({interval, combined});

        return {interval, combined};
    }

private:
    BeamCorrections<S> beamcorrections;
    std::optional<PhaseCorrections<S>> phasecorrections;
    std::unordered_map<int, std::vector<std::tuple<Interval, aterm_t>>> aterms {};
};

} // namespace Aterms