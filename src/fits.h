#pragma once

#include <array>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include <fitsio.h>
#include <wcslib/wcslib.h>

#include "memory.h"
#include "outputtypes.h"

namespace fits {

template <typename S>
int getbitpix();

template <>
int getbitpix<float>() { return FLOAT_IMG; }

template <>
int getbitpix<double>() { return DOUBLE_IMG; }

template <typename S>
int getfitsdatatype();

template <>
int getfitsdatatype<float>() { return TFLOAT; }

template <>
int getfitsdatatype<double>() { return TDOUBLE; }

template <typename S> requires(std::is_floating_point<S>::value)
void save(
    std::string_view fname, HostSpan<S, 2>&arr, GridSpec gridspec, RaDec phasecenter
) {
    // Append '!' to filename to indicate to cfitsio to overwrite existing file
    std::string fpath("!");
    fpath += fname;

    int status {};
    fitsfile* fptr {};
    fits_create_file(&fptr, fpath.c_str(), &status);

    std::array<long, 2> naxes {
        static_cast<long>(arr.size(0)), static_cast<long>(arr.size(1))
    };
    fits_create_img(fptr, getbitpix<S>(), 2, naxes.data(), &status);

    // Add WCS information
    {
        // wcserr_enable() is not threadsafe, but the static assignment ensures it is
        // called just once, without race condition
        [[maybe_unused]] static int _ = wcserr_enable(true);

        wcsprm wcs;
        wcs.flag = -1;
        if (wcsinit(true, 2, &wcs, 0, 0, 0) != 0) {
            auto msg = fmt::format(
                "An error occurred creating a FITS file: {}", wcs.err->msg
            );
            throw std::runtime_error(msg);
        }

        // Add wcs configuration
        wcs.flag = 0;
        wcs.crval[0] = phasecenter.ra;
        wcs.crval[1] = phasecenter.dec;
        wcs.crpix[0] = (gridspec.Nx / 2) - gridspec.deltalpx + 1; // 1-indexed
        wcs.crpix[1] = (gridspec.Ny / 2) - gridspec.deltampx + 1; // 1-indexed
        wcs.cdelt[0] = -std::asin(gridspec.scalel);
        wcs.cdelt[1] = std::asin(gridspec.scalem);
        std::strncpy(wcs.cunit[0], "rad", 72);
        std::strncpy(wcs.cunit[1], "rad", 72);
        std::strncpy(wcs.ctype[0], "RA---SIN", 72);
        std::strncpy(wcs.ctype[1], "DEC--SIN", 72);

        if (wcsset(&wcs) != 0) {
            throw std::runtime_error(wcs.err->msg);
        };

        // Create headers
        int nkeyrec;
        char* hdrs;
        if (wcshdo(WCSHDO_none, &wcs, &nkeyrec, &hdrs) != 0) {
            auto msg = fmt::format(
                "An error occurred creating a FITS file: {}", wcs.err->msg
            );
            throw std::runtime_error(msg);
        }

        // Write out to the fits file
        for (int i {}; i < nkeyrec; ++i) {
            fits_write_record(fptr, &hdrs[i * 80], &status);
        }

        // Free the wcs memory allocations
        wcsfree(&wcs);
        wcsdealloc(hdrs);
    }

    std::array<long, 2> fpixel {1, 1};
    fits_write_pix(
        fptr, getfitsdatatype<S>(), fpixel.data(),
        static_cast<long long>(arr.size()),  arr.data(), &status
    );

    fits_close_file(fptr, &status);

    if (status != 0) {
        char fitsmsg[FLEN_STATUS] {};
        fits_get_errstatus(status, fitsmsg);

        auto msg = fmt::format("An error occurred creating a FITS file: {}", fitsmsg);
        throw std::runtime_error(msg);
    }
}

template <typename S>
void save(
    std::string_view fname, HostSpan<StokesI<S>, 2> stokesI,
    GridSpec gridspec, RaDec phasecenter
) {
    HostArray<S, 2> stokesIreal {stokesI.shape()};
    for (size_t i {}; i < stokesI.size(); ++i) {
        stokesIreal[i] = stokesI[i].I.real();
    }
    save(fname, stokesIreal, gridspec, phasecenter);
}

template <typename S>
void save(
    std::string_view fname, HostSpan<thrust::complex<S>, 2> img,
    GridSpec gridspec, RaDec phasecenter
) {
    HostArray<S, 2> imgreal {img.shape()};
    for (size_t i {}; i < img.size(); ++i) {
        imgreal[i] = img[i].real();
    }
    save(fname, imgreal, gridspec, phasecenter);
}

template <typename S, int N>
requires (std::is_same_v<S, float> || std::is_same_v<S, double>)
HostArray<S, N> open(const std::string& path) {
    // Open the file
    int status {};
    fitsfile* fptr {};
    fits_open_image(&fptr, path.c_str(), READONLY, &status);

    // Get file metadata
    int bitpix, naxis;
    std::array<long, N> naxes;
    fits_get_img_param(fptr, N,  &bitpix, &naxis, naxes.data(), &status);

    if (status !=  0) {
        char fitsmsg[FLEN_STATUS] {};
        fits_get_errstatus(status, fitsmsg);

        auto msg = fmt::format("An error occurred opening {}: {}", path, fitsmsg);
        throw std::runtime_error(msg);
    }

    // Validate file metadata fits our expectations
    if (naxis != N) throw std::runtime_error(fmt::format(
        "Opening {} failed: expected image with {} dimensions, got {}", path, N, naxis
    ));

    if (bitpix != -32 && bitpix != -64) throw std::runtime_error(fmt::format(
        "Opening {} failed: expected image with single or double precision data, "
        " got {}-bit integer data instead", path, bitpix
    ));

    // Create data array with axes that match FITS file
    std::array<long long, N> dims;
    for (size_t i {}; long n : naxes) dims[i++] = n;
    HostArray<S, N> data(dims);

    // Read data into array
    int datatype = std::is_same_v<S, float> ? TFLOAT : TDOUBLE;
    std::array<long, N> fpixel;
    fpixel.fill(1);  // 1-indexed
    long nelements = 1;
    for (long n : naxes) nelements *= n;
    S nulval {};
    int anynul;

    fits_read_pix(
        fptr, datatype, fpixel.data(), nelements, &nulval,
        data.data(), &anynul, &status
    );

    if (status != 0) {
        char fitsmsg[FLEN_STATUS] {};
        fits_get_errstatus(status, fitsmsg);

        auto msg = fmt::format("An error occurred opening {}: {}", path, fitsmsg);
        throw std::runtime_error(msg);
    }

    if (nulval != 0) throw std::runtime_error(fmt::format(
        "Opening {} failed: NULL values detected in data", path
    ));

    return data;
}

}  // namespace fits