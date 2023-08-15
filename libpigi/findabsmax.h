#pragma once

#include <complex>
#include <tuple>

template <typename T>
std::tuple<size_t, std::complex<T>> findabsmax(std::complex<T>* first, std::complex<T>* last);