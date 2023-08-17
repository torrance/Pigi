#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <tuple>
#include <type_traits>

#include <fmt/format.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include "gsl/gsl_multifit_nlinear.h"
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "fft.h"
#include "gridspec.h"
#include "hip.h"
#include "memory.h"
#include "outputtypes.h"
#include "util.h"

namespace clean {

template <typename T, typename S>
__global__ void _subtractpsf(
    DeviceSpan<T, 2> img, const GridSpec imgGridspec,
    const DeviceSpan<T, 2> psf, const GridSpec psfGridspec,
    long long xpeak, long long ypeak, S f
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < psfGridspec.size();
        idx += blockDim.x * gridDim.x
    ) {
        auto [xpx, ypx] = psfGridspec.linearToGrid(idx);

        // Set origin to center of PSF
        xpx -= static_cast<long long>(psfGridspec.Nx) / 2;
        ypx -= static_cast<long long>(psfGridspec.Ny) / 2;

        // Set origin to bottom left corner of img
        xpx += static_cast<long long>(imgGridspec.Nx) / 2;
        ypx += static_cast<long long>(imgGridspec.Ny) / 2;

        long long xoffset { xpeak - static_cast<long long>(imgGridspec.Nx) / 2 };
        long long yoffset { ypeak - static_cast<long long>(imgGridspec.Ny) / 2 };

        // Now shift based on location of peak
        xpx += xoffset;
        ypx += yoffset;

        if (
            0 <= xpx && xpx < static_cast<long long>(imgGridspec.Nx) &&
            0 <= ypx && ypx < static_cast<long long>(imgGridspec.Ny)
        ) {
            auto cell = psf[idx];
            img[imgGridspec.gridToLinear(xpx, ypx)] -= (cell *= f);
        }
    }
}

template <typename T, typename S>
void subtractpsf(
    DeviceArray<T, 2>& img, const GridSpec imgGridspec,
    const DeviceArray<T, 2>& psf, const GridSpec psfGridspec,
    long long xpeak, long long ypeak, S f
) {
    auto fn = _subtractpsf<T, S>;
    auto [nblocks, nthreads] = getKernelConfig(
        fn, psfGridspec.size()
    );
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        img.asSpan(), imgGridspec, psf.asSpan(), psfGridspec, xpeak, ypeak, f
    );
}

struct Config {
    double gain {0.1};
    double mgain {0.8};
    double threshold {0};
    size_t niter {std::numeric_limits<size_t>::max()};
};

template <typename S>
std::tuple<HostArray<StokesI<S>, 2>, size_t> major(
    HostArray<StokesI<S>, 2>& img,
    const GridSpec imgGridspec,
    const HostArray<StokesI<S>, 2>& psf,
    const GridSpec psfGridspec,
    const Config config
) {
    HostArray<StokesI<S>, 2> components({img.size(0), img.size(1)});

    // Clean down to either:
    //   1. the explicit threshold limit, or
    //   2. the current peak value minus the mgain
    // (whichever is greater).
    S maxInit {};
    for (auto& val : img) {
        maxInit = std::max(maxInit, std::abs(val.real()));
    }

    auto threshold = std::max((1 - config.mgain) * maxInit, config.threshold);
    fmt::println(
        "Beginning major clean cycle: from {:.2g} Jy to {:.2g}", maxInit, threshold)
    ;

    // Transfer img and psf to device
    DeviceArray<StokesI<S>, 2> img_d(img);
    DeviceArray<StokesI<S>, 2> psf_d(psf);

    size_t iter {};
    while (++iter < config.niter) {
        // Find the device pointer to maximum value
        StokesI<S>* maxptr = thrust::max_element(
            thrust::device, img_d.begin(), img_d.end(), [](auto lhs, auto rhs) {
                return std::abs(lhs.I.real()) < std::abs(rhs.I.real());
            }
        );
        size_t idx = maxptr - img_d.begin();

        // Copy max value host -> device
        StokesI<S> maxval;
        HIPCHECK(
            hipMemcpyDtoHAsync(&maxval, maxptr, sizeof(StokesI<S>), hipStreamPerThread)
        );
        HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );

        // Apply gain
        auto val = maxval.real() * static_cast<S>(config.gain);
        auto [xpx, ypx] = imgGridspec.linearToGrid(idx);

        // Save component and subtract contribution from image
        components[idx] += val;
        subtractpsf<StokesI<S>, S>(
            img_d, imgGridspec, psf_d, psfGridspec, xpx, ypx, val
        );

        if (iter % 1000 == 0) fmt::println(
            "   [{} iteration] {:.2g} Jy peak found", iter, std::abs(maxval.I)
        );

        if (std::abs(maxval.I) <= threshold) break;
    }

    img = img_d;

    maxInit = 0;
    for (auto& val : img) {
        maxInit = std::max(maxInit, std::abs(val.real()));
    }
    fmt::println(
        "Clean cycle complete ({} iterations). Peak value remaining: {:.2g} Jy",
        iter, maxInit
    );

    return std::make_tuple(std::move(components), iter);
}

int gaussian(const gsl_vector* params, void* data, gsl_vector* residual) {
    auto dirtypsf = static_cast<HostSpan<double, 2>*>(data);
    GridSpec gridspec {dirtypsf->size(0), dirtypsf->size(1), 0, 0};

    // Get current model params
    double xsigma = gsl_vector_get(params, 0);
    double ysigma = gsl_vector_get(params, 1);
    double pa = gsl_vector_get(params, 2);

    for (size_t idx {}; idx < dirtypsf->size(); ++idx) {
        auto [xpx, ypx] = gridspec.linearToGrid(idx);
        xpx -= static_cast<long long>(dirtypsf->size(0) / 2);
        ypx -= static_cast<long long>(dirtypsf->size(1) / 2);

        // Rotate xpx, ypx by position angle
        double x {xpx * std::cos(pa) - ypx * std::sin(pa)};
        double y {xpx * std::sin(pa) + ypx * std::cos(pa)};

        double model = std::exp(
            -(x * x) / (2 * xsigma * xsigma) - (y * y) / (2 * ysigma * ysigma)
        );

        gsl_vector_set(residual, idx, model - (*dirtypsf)[idx]);
    };

    return GSL_SUCCESS;
}

struct PSF {
    // All values in radians
    double major;
    double minor;
    double pa;

    template <typename T>
    HostArray<T, 2> draw(GridSpec gridspec) {
        HostArray<T, 2> psf({gridspec.Nx, gridspec.Ny});

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
};

PSF fitpsf(const HostArray<double, 2>& dirtypsf, const GridSpec gridspec) {
    // Linear fit method
    const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;

    // Create parameter vector and provide initial guess
    std::array<double, 3> params0 {5, 5, 0};
    gsl_vector_view params = gsl_vector_view_array(params0.data(), params0.size());

    // Allocate the workspace with psf.size() space and 3 parameters to solve
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    gsl_multifit_nlinear_workspace* w = gsl_multifit_nlinear_alloc(
        T, &fdf_params, dirtypsf.size(), params0.size()
    );

    // Define the function to be minimised
    gsl_multifit_nlinear_fdf fdf;
    fdf.f = gaussian;
    fdf.df = NULL; // using default finite-difference Jacobian
    fdf.n = dirtypsf.size();
    fdf.p = params0.size(); // 3 parameters to solve for
    fdf.params = (void*) &dirtypsf;

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
    PSF psf {
        f * gsl_vector_get(w->x, 0),
        f * gsl_vector_get(w->x, 1),
        std::fmod(gsl_vector_get(w->x, 2), 2 * ::pi_v<double>)
    };

    // Free
    gsl_multifit_nlinear_free(w);

    fmt::println(
        "Gaussian fit: {:.2f}' x {:.2f}' (pa {:.1f} degrees)",
        rad2deg(psf.major) * 60, rad2deg(psf.minor) * 60, rad2deg(psf.pa)
    );

    return psf;
}

template <typename T, typename S>
void convolve(HostArray<T, 2>& img, const HostArray<S, 2>& kernel) {
    shapecheck(img, kernel);

    // Pad the images
    long long xpadding {img.size(0) / 2};
    long long ypadding {img.size(1) / 2};
    HostArray<T, 2> img_padded(
        {2 * img.size(0), 2 * img.size(1)}
    );
    HostArray<S, 2> kernel_padded(
        {2 * kernel.size(0), 2 * kernel.size(1)}
    );

    GridSpec gridspec {img.size(0), img.size(1), 0, 0};
    GridSpec gridspec_padded {img_padded.size(0), img_padded.size(1), 0, 0};

    for (long long nx {0}; nx < img.size(0); ++nx) {
        for (long long ny {0}; ny < img.size(1); ++ny) {
            size_t idxSrc = gridspec.gridToLinear(nx, ny);
            size_t idxDst = gridspec_padded.gridToLinear(
                nx + xpadding, ny + ypadding
            );

            img_padded[idxDst] = img[idxSrc];
            kernel_padded[idxDst] = kernel[idxSrc];
        }
    }

    // Create fft plans
    auto planImg = fftPlan<T>(gridspec_padded);
    auto planKernel = fftPlan<S>(gridspec_padded);

    // Send to device
    DeviceArray<T, 2> img_d {img_padded};
    DeviceArray<S, 2> kernel_d {kernel_padded};

    // FT forward
    fftExec(planImg, img_d, HIPFFT_FORWARD);
    fftExec(planKernel, kernel_d, HIPFFT_FORWARD);

    // Multiply in FT domain and normalize
    img_d.mapInto([=] __device__ (auto img, auto kernel) {
        return (img *= kernel) /= gridspec_padded.size();
    }, img_d.asSpan(), kernel_d.asSpan());

    // FT backward
    fftExec(planImg, img_d, HIPFFT_BACKWARD);

    // Copy back from device
    img_padded = img_d;

    img = crop(img_padded, xpadding, ypadding);
}

}