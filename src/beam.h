#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>

#include <mwa_hyperbeam.h>

#include "coordinates.h"
#include "gridspec.h"
#include "memory.h"
#include "mset.h"
#include "outputtypes.h"
#include "util.h"

namespace Beam {

template<typename Q>
class Beam {
public:
    virtual ComplexLinearData<Q> pointResponse(const RaDec& azel, const double freq) = 0;

    virtual HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const RaDec& origin, const double freq
    ) = 0;

    virtual ~Beam() = default;
};

template <typename Q>
class Uniform : public Beam<Q> {
public:
    Uniform() {}

    ComplexLinearData<Q> pointResponse(const RaDec&, const double) override {
        return {1, 0, 0, 1};
    }

    HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const RaDec&, const double
    ) override {
        HostArray<ComplexLinearData<Q>, 2> beam {gridspec.Nx, gridspec.Ny};
        beam.fill({1, 0, 0, 1});
        return beam;
    }
};

template <typename Q>
class Gaussian : public Beam<Q> {
public:
    Gaussian(RaDec origin = {0, 0}, double sigma = 0) : origin(origin), sigma(sigma) {}

    ComplexLinearData<Q> pointResponse(const RaDec& radec, const double) override {
        // Calculate angular offset from GaussianBeam origin using Haversine formula
        auto theta = 2 * std::asin(
            std::sqrt(
                std::pow(std::sin((radec.dec - origin.dec) / 2), 2) +
                std::cos(radec.dec) * std::cos(origin.dec) *
                std::pow(std::sin((radec.ra - origin.ra) / 2), 2)
            )
        );

        // Just a regular circular gaussian
        // We take the square root because the power is J * J^H
        auto power = std::sqrt(
            std::exp(-(theta * theta) / (2 * sigma * sigma))
        );

        return {power, 0, 0, power};
    }

    HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const RaDec& gridorigin, const double freq
    ) override {

        HostArray<ComplexLinearData<Q>, 2> beam {gridspec.Nx, gridspec.Ny};

        for (size_t i {}; i < gridspec.size(); ++ i) {
            auto [l, m] = gridspec.linearToSky<double>(i);
            auto radec = lmToRaDec(l, m, gridorigin);
            beam[i] = pointResponse(radec, freq);
        }

        return beam;
    }

private:
    RaDec origin;
    double sigma;
};

template <typename Q>
class MWA : public Beam<Q> {
public:
    MWA(
        const double mjd = {},
        const std::array<uint32_t, 16>& delays = {},
        const std::array<double, 16>& amps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    ) :  mjd(mjd), delays(delays), amps(amps) {

        // Allocate FEE beam or throw error
        auto err = new_fee_beam_from_env(&feebeam);
        if (err != 0) {
            int length = hb_last_error_length();
            std::string msg(length, ' ');
            hb_last_error_message(msg.data(), length);
            throw std::runtime_error(msg);
        }
    }

    MWA(const MWA&) = delete;
    MWA(MWA&& other) {
        *this = std::move(other);
    }

    MWA& operator=(const MWA& other) = delete;
    MWA& operator=(MWA&& other) {
        using std::swap;
        swap(mjd, other.mjd);
        swap(delays, other.delays);
        swap(amps, other.amps);
        swap(feebeam, other.feebeam);
        return *this;
    }

    ~MWA() override {
        if (feebeam) free_fee_beam(feebeam);
        feebeam = {};
    }

    ComplexLinearData<Q> pointResponse(const RaDec& radec, double freq) override {
        auto azel = radecToAzel(radec, mjd, origin);

        ComplexLinearData<double> jones;

        auto err = fee_calc_jones(
            feebeam,
            azel.az,
            pi_v<double> / 2 - azel.el,
            freq,
            delays.data(),
            amps.data(),
            16,  // Number of amps (can be 16 or 32)
            true, // Norm to zenith
            &origin.lat,
            false, // Don't use IAU order; use MWA order
            reinterpret_cast<double*>(&jones)
        );

        if (err != 0) {
            int length = hb_last_error_length();
            std::string msg(length, ' ');
            hb_last_error_message(msg.data(), length);
            throw std::runtime_error(msg);
        }

        return static_cast<ComplexLinearData<Q>>(jones);
    }

    HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const RaDec& gridorigin, const double freq
    ) override {
        // Create vectors of az and za to pass to hyperbeam
        std::vector<double> azs, zas;
        for (size_t i {}; i < gridspec.size(); ++ i) {
            auto [l, m] = gridspec.linearToSky<double>(i);
            auto azel = radecToAzel(lmToRaDec(l, m, gridorigin), mjd, origin);
            azs.push_back(azel.az);
            zas.push_back(pi_v<double> / 2 - azel.el);
        }

        // Allocate double precision array
        HostArray<ComplexLinearData<double>, 2> beam {gridspec.Nx, gridspec.Ny};

        auto err = fee_calc_jones_array(
            feebeam,
            gridspec.size(),
            azs.data(),
            zas.data(),
            freq,
            delays.data(),
            amps.data(),
            16,  // Number of amps (can be 16 or 32)
            true, // Norm to zenith
            &origin.lat,
            false, // Don't use IAU order; use MWA order
            reinterpret_cast<double*>(beam.data())
        );

        if (err != 0) {
            int length = hb_last_error_length();
            std::string msg(length, ' ');
            hb_last_error_message(msg.data(), length);
            throw std::runtime_error(msg);
        }

        // Return beam at required precision; cast only runs if precision
        // differs from double
        return static_cast<HostArray<ComplexLinearData<Q>, 2>>(beam);
    }

    static constexpr LongLat origin {
        .lon = deg2rad(116. + 40. / 60. + 14.93 / 3600.),
        .lat = -deg2rad(26. + 42. / 60. + 11.95 / 3600.)
    };

private:
    double mjd {};
    std::array<uint32_t, 16> delays {};
    std::array<double, 16> amps {};
    FEEBeam* feebeam {};
};

}
