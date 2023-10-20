#pragma once

#include <array>
#include <cmath>

#include <fmt/format.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <thrust/complex.h>

#include "gridspec.h"
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
        // Linear fit method
        const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;

        // Create parameter vector and provide initial guess
        std::array<double, 3> params0 {5, 5, 0};
        gsl_vector_view params = gsl_vector_view_array(params0.data(), params0.size());

        // Allocate the workspace with psf.size() space and 3 parameters to solve
        gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
        gsl_multifit_nlinear_workspace* w = gsl_multifit_nlinear_alloc(
            T, &fdf_params, psfDirty.size(), params0.size()
        );

        // Define the function to be minimised
        gsl_multifit_nlinear_fdf fdf;
        fdf.f = PSF::minimizationFn;
        fdf.df = NULL; // using default finite-difference Jacobian
        fdf.n = psfDirty.size();
        fdf.p = params0.size(); // 3 parameters to solve for
        fdf.params = const_cast<void*>(
            static_cast<const void*>(&psfDirty)
        );

        // Initialize the workspace with function and initial params guess
        int status;
        status = gsl_multifit_nlinear_init(&params.vector, &fdf, w);
        if (status != GSL_SUCCESS) {
            fmt::println(stderr, "An error occurred initializing the GSL workspace");
            abort();
        }

        fmt::println("Fitting PSF with model Gaussian...");
        int info;
        status = gsl_multifit_nlinear_driver(
            100, // maxiter
            1e-8, // xtol
            1e-8, // std::pow(GSL_DBL_EPSILON, 1./3), // gtol
            1e-8, // ftol
            NULL, // callback
            NULL, // callback params
            &info,
            w // workspace
        );

        switch (status) {
        case GSL_SUCCESS:
            switch (info) {
            case 1:
                fmt::println("PSF fit converged (reason: small step size)");
                break;
            case 2:
                fmt::println("PSF fit converged (reason: small gradient)");
                break;
            default:
                // This is meant to be unreachable
                fmt::println(stderr, "PSF fit returned success but info not set correctly");
                abort();
            }
            break;
        case GSL_EMAXITER:
            fmt::println(stderr, "Warning: PSF fit reached max iterations before convergence");
            break;
        case GSL_ENOPROG:
            fmt::println(
                stderr, "Warning PSF did not converge (reason: no new acceptable delta could be found)"
            );
            break;
        default:
            fmt::println(stderr, "PSF fit returned unknown error");
            abort();
        }

        // Convert psf xsigma and ysigma to FWHM angular values
        double f = 2 * std::sqrt(2 * std::log(2)) * std::asin(gridspec.scalelm);
        major = f * gsl_vector_get(w->x, 0);
        minor = f * gsl_vector_get(w->x, 1);
        pa = std::fmod(gsl_vector_get(w->x, 2), 2 * ::pi_v<double>);

        // Free
        gsl_multifit_nlinear_free(w);

        fmt::println(
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
                fmt::println(
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
                fmt::println(
                    "Cropping PSF from {}x{} to {}x{}",
                    gridspec.Nx, gridspec.Ny,
                    gridspecWindowed.Nx, gridspecWindowed.Ny)
                ;
                return gridspecWindowed;
            }
        }
    }

    fmt::println(
        stderr,
        "An error occurred attempting to window PSF; using full {}x{} image (slower)",
        gridspec.Nx, gridspec.Ny
    );
    return gridspec;
};