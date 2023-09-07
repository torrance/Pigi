#pragma once

#include <cmath>
#include <stdexcept>

#include <mwa_hyperbeam.h>

#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"

struct AzEl {
    double az;  // Azimuth, measured North through to East (radians)
    double el;  // Elevation angle, from horizon (radians)
};

AzEl lmToAzEl(const double l, const double m, const AzEl origin) {
    // Don't use ndash here, as we _want_ below horizon values to be invalid
    // and for NaN to propagate
    auto n = std::sqrt(1 - l * l - m * m);

    auto az = origin.az + std::atan2(l, n * std::cos(origin.el) - m * std::sin(origin.el));
    auto el = std::asin(m * std::cos(origin.el) + n * std::sin(origin.el));

    return {az, el};
}

namespace Beam {

template<typename Q>
class Beam {
public:
    virtual ComplexLinearData<Q> pointResponse(const AzEl& azel, const double freq) = 0;

    virtual HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const AzEl& origin, const double freq
    ) = 0;

    virtual ~Beam() = default;
};

template <typename Q>
class Uniform : public Beam<Q> {
public:
    Uniform() {}

    ComplexLinearData<Q> pointResponse(const AzEl&, const double) override {
        return {1, 0, 0, 1};
    }

    HostArray<ComplexLinearData<Q>, 2> gridResponse(
        const GridSpec& gridspec, const AzEl&, const double
    ) override {
        HostArray<ComplexLinearData<Q>, 2> beam {gridspec.Nx, gridspec.Ny};
        beam.fill({1, 0, 0, 1});
        return beam;
    }
};

template <typename Q>
class Gaussian : public Beam<Q> {
public:
    Gaussian(AzEl origin = {0, pi_v<double> / 2}, double sigma = 0) : origin(origin), sigma(sigma) {}

    ComplexLinearData<Q> pointResponse(const AzEl& azel, const double) override {
        // Calculate angular offset from GaussianBeam origin using Haversine formula
        auto theta = 2 * std::asin(
            std::sqrt(
                std::pow(std::sin((azel.el - origin.el) / 2), 2) +
                std::cos(azel.el) * std::cos(origin.el) *
                std::pow(std::sin((azel.az - origin.az) / 2), 2)
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
        const GridSpec& gridspec, const AzEl& gridorigin, const double freq
    ) override {

        HostArray<ComplexLinearData<Q>, 2> beam {gridspec.Nx, gridspec.Ny};

        for (size_t i {}; i < gridspec.size(); ++ i) {
            auto [l, m] = gridspec.linearToSky<double>(i);
            auto azel = lmToAzEl(l, m, gridorigin);
            beam[i] = pointResponse(azel, freq);
        }

        return beam;
    }

private:
    AzEl origin;
    double sigma;
};

template <typename Q>
class MWA : public Beam<Q> {
public:
    MWA() {
        // Allocate FEE beam or throw error
        auto err = new_fee_beam_from_env(&feebeam);
        if (err != 0) {
            int length = hb_last_error_length();
            std::string msg(length, ' ');
            hb_last_error_message(msg.data(), length);
            throw std::runtime_error(msg);
        }
    }

    MWA(const std::array<uint32_t, 16>& delays) : delays(delays), MWA() {}

    MWA(const std::array<uint32_t, 16>& delays, const std::array<double, 16>& amps) :
        delays(delays), amps(amps), MWA() {}

    MWA(const MWA&) = delete;
    MWA(MWA&& other) {
        *this = std::move(other);
    }

    MWA& operator=(const MWA& other) = delete;
    MWA& operator=(MWA&& other) {
        using std::swap;
        swap(delays, other.delays);
        swap(amps, other.amps);
        swap(feebeam, other.feebeam);
        return *this;
    }

    ~MWA() override {
        if (feebeam) free_fee_beam(feebeam);
        feebeam = {};
    }

    ComplexLinearData<Q> pointResponse(const AzEl& azel, double freq) override {
        ComplexLinearData<double> jones;

        auto err = calc_jones(
            feebeam,
            azel.az,
            pi_v<double> / 2 - azel.el,
            freq,
            delays.data(),
            amps.data(),
            16,  // Number of amps (can be 16 or 32)
            true, // Norm to zenith
            &latitude,
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
        const GridSpec& gridspec, const AzEl& gridorigin, const double freq
    ) override {

        // Create vectors of az and za to pass to hyperbeam
        std::vector<double> azs, zas;
        for (size_t i {}; i < gridspec.size(); ++ i) {
            auto [l, m] = gridspec.linearToSky<double>(i);
            auto azel = lmToAzEl(l, m, gridorigin);
            azs.push_back(azel.az);
            zas.push_back(pi_v<double> / 2 - azel.el);
        }

        // Allocate double precision array
        HostArray<ComplexLinearData<double>, 2> beam_double {gridspec.Nx, gridspec.Ny};

        auto err = calc_jones_array(
            feebeam,
            gridspec.size(),
            azs.data(),
            zas.data(),
            freq,
            delays.data(),
            amps.data(),
            16,  // Number of amps (can be 16 or 32)
            true, // Norm to zenith
            &latitude,
            false, // Don't use IAU order; use MWA order
            reinterpret_cast<double*>(beam_double.data())
        );

        if (err != 0) {
            int length = hb_last_error_length();
            std::string msg(length, ' ');
            hb_last_error_message(msg.data(), length);
            throw std::runtime_error(msg);
        }

        // Cast to Q precision if needed; else return the beam
        if constexpr(std::is_same<Q, double>::value) {
            return beam_double;
        }

        HostArray<ComplexLinearData<Q>, 2> beam {gridspec.Nx, gridspec.Ny};
        for (size_t i {}; i < gridspec.size(); ++i) {
            beam[i] = static_cast<ComplexLinearData<Q>>(beam_double[i]);
        }
        return beam;
    }

private:
    std::array<uint32_t, 16> delays {};
    std::array<double, 16> amps = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    FEEBeam* feebeam {};

    // MWA lattitude: −26°42′11′′.94986
    static constexpr double latitude = -deg2rad(26 * 42./60 + 11.94986/3600);
};

}
