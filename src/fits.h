#pragma once

#include <array>
#include <string_view>

#include <fitsio.h>

#include "memory.h"
#include "outputtypes.h"

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
void save(std::string_view fname, HostArray<S, 2>& arr) {
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

    std::array<long, 2> fpixel {1, 1};
    fits_write_pix(
        fptr, getfitsdatatype<S>(), fpixel.data(),
        static_cast<long long>(arr.size()),  arr.data(), &status
    );

    fits_close_file(fptr, &status);

    if (status != 0) {
        fits_report_error(stderr, status);
        abort();
    }
}

template <typename S>
void save(std::string_view fname, HostArray<StokesI<S>, 2>& stokesI) {
    HostArray<S, 2> stokesIreal(stokesI.shape());
    for (size_t i {}; i < stokesI.size(); ++i) {
        stokesIreal[i] = stokesI[i].I.real();
    }
    save(fname, stokesIreal);
}

template <typename S>
void save(std::string_view fname, HostArray<std::complex<S>, 2>& img) {
    HostArray<S, 2> imgreal(img.shape());
    for (size_t i {}; i < img.size(); ++i) {
        imgreal[i] = img[i].real();
    }
    save(fname, imgreal);
}