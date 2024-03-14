#pragma once

#include <array>
#include <cmath>

#include <fmt/format.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <thrust/complex.h>

#include "gslfit.h"
#include "gridspec.h"
#include "logger.h"
#include "memory.h"
#include "util.h"

template <typename P>
class PSF {
public:
    // All values in radians
    double major {};
    double minor {};
    double pa {};

    PSF() = default;

    PSF(double major, double minor, double pa) : major(major), minor(minor), pa(pa) {}

    PSF(const HostSpan<thrust::complex<P>, 2> psfDirty, const GridSpec gridspec) {
        GSLFit fitter(PSF::minimizationFn, 3, psfDirty.size());
        auto params = fitter.fit(
            {5, 5, 0}, const_cast<void*>(static_cast<const void*>(&psfDirty))
        );

        // Convert psf xsigma and ysigma to FWHM angular values
        double f = 2 * std::sqrt(2 * std::log(2)) * std::asin(gridspec.scalelm);
        major = f * params[0];
        minor = f * params[1];
        pa = std::fmod(params[2], 2 * ::pi_v<double>);

        Logger::info(
            "PSF with Gaussian fit: {:.2f}' x {:.2f}' (pa {:.1f} degrees)",
            rad2deg(major) * 60, rad2deg(minor) * 60, rad2deg(pa)
        );
    }

    HostArray<thrust::complex<P>, 2> draw(GridSpec gridspec) {
        HostArray<thrust::complex<P>, 2> psf {gridspec.Nx, gridspec.Ny};

        // Convert major, minor from radians to pixels
        // and FWHM to sigma
        double f = 1 / (2 * std::sqrt(2 * std::log(2)) * std::asin(gridspec.scalelm));
        double xsigma {f * major};
        double ysigma {f * minor};

        for (size_t idx {}; idx < psf.size(); ++idx) {
            auto [xpx, ypx] = gridspec.linearToGrid(idx);
            xpx -= static_cast<long long>(psf.size(0) / 2);
            ypx -= static_cast<long long>(psf.size(1) / 2);

            // Rotate xpx, ypx by position angle
            double x {xpx * std::cos(pa) - ypx * std::sin(pa)};
            double y {xpx * std::sin(pa) + ypx * std::cos(pa)};

            psf[idx] = std::exp(
                -(x * x) / (2 * xsigma * xsigma) - (y * y) / (2 * ysigma * ysigma)
            );
        }

        return psf;
    }

private:
    /**
     * Compute the difference between Gaussian with params and the _real_ components
     * of the psfDirty
     */
    static int minimizationFn(const gsl_vector* params, void* data, gsl_vector* residual) {
        auto psfDirty = static_cast<HostSpan<thrust::complex<P>, 2>*>(data);
        GridSpec gridspec {psfDirty->size(0), psfDirty->size(1), 0, 0};

        // Get current model params
        double xsigma = gsl_vector_get(params, 0);
        double ysigma = gsl_vector_get(params, 1);
        double pa = gsl_vector_get(params, 2);

        for (size_t idx {}; idx < psfDirty->size(); ++idx) {
            auto [xpx, ypx] = gridspec.linearToGrid(idx);

            // Set origin to centre pixel
            xpx -= psfDirty->size(0) / 2;
            ypx -= psfDirty->size(1) / 2;

            // Rotate xpx, ypx by position angle
            double x {xpx * std::cos(pa) - ypx * std::sin(pa)};
            double y {xpx * std::sin(pa) + ypx * std::cos(pa)};

            double model = std::exp(
                -(x * x) / (2 * xsigma * xsigma) - (y * y) / (2 * ysigma * ysigma)
            );

            gsl_vector_set(residual, idx, model - (*psfDirty)[idx].real());
        };

        return GSL_SUCCESS;
    }
};

template <typename P>
auto cropPsf(HostSpan<thrust::complex<P>, 2> psfDirty, GridSpec gridspec, double threshold) {
    for (long long edge {}; edge < std::min(psfDirty.size(0), psfDirty.size(1)) / 2; ++edge) {
        // Prepare windowed gridspec to use for resize()
        auto gridspecWindowed = GridSpec::fromScaleLM(
            psfDirty.size(0) - 2 * std::max(edge - 1, 0ll),
            psfDirty.size(1) - 2 * std::max(edge - 1, 0ll),
            gridspec.scalelm
        );

        // Top and bottom
        for (long long nx {edge}; nx < psfDirty.size(0) - edge; ++nx) {
            auto idx1 = gridspec.gridToLinear(nx, edge);
            auto idx2 = gridspec.gridToLinear(nx, psfDirty.size(1) - edge - 1);
            if (
                std::abs(psfDirty[idx1].real()) > threshold ||
                std::abs(psfDirty[idx2].real()) > threshold
            ) {
                Logger::info(
                    "Cropping PSF from {}x{} to {}x{}",
                    gridspec.Nx, gridspec.Ny,
                    gridspecWindowed.Nx, gridspecWindowed.Ny)
                ;
                return gridspecWindowed;
            }
        }

        // Left and right
        for (long long ny {edge}; ny < psfDirty.size(1) - edge; ++ny) {
            auto idx1 = gridspec.gridToLinear(edge, ny);
            auto idx2 = gridspec.gridToLinear(psfDirty.size(0) - edge - 1, ny);
            if (
                std::abs(psfDirty[idx1].real()) > threshold ||
                std::abs(psfDirty[idx2].real()) > threshold
            ) {
                Logger::info(
                    "Cropping PSF from {}x{} to {}x{}",
                    gridspec.Nx, gridspec.Ny,
                    gridspecWindowed.Nx, gridspecWindowed.Ny)
                ;
                return gridspecWindowed;
            }
        }
    }

    Logger::warning(
        "An error occurred attempting to window PSF; using full {}x{} image (slower)",
        gridspec.Nx, gridspec.Ny
    );
    return gridspec;
};