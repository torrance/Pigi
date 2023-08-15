#pragma once

#include <cmath>
#include <tuple>
#include <type_traits>

#include "fft.h"
#include "gridspec.h"
#include "memory.h"
#include "outputtypes.h"

namespace clean {

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
    const Config config = Config()
);

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

PSF fitpsf(const HostArray<double, 2>& dirtypsf, const GridSpec gridspec);

template <typename T, typename S>
void convolve(HostArray<T, 2>& img, const HostArray<S, 2>& kernel);

}