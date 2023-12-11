#pragma once

#include <span>
#include <type_traits>
#include <vector>

#include <gsl/gsl_multifit.h>

template <typename P1, typename P2> requires(
    std::is_floating_point_v<P1>, std::is_floating_point_v<P2>
)
std::vector<double> polyfit(std::span<P1> xs, std::span<P2> ys, int roots) {

    // Create X matrix
    auto X = gsl_matrix_alloc(xs.size(), roots);
    auto Y = gsl_vector_alloc(ys.size());

    for (size_t i {}; i < xs.size(); ++i) {
        gsl_vector_set(Y, i, ys[i]);

        for (int j {}; j < roots; ++j) {
            gsl_matrix_set(X, i, j, std::pow(xs[i], j));
        }
    }

    double chisq {};
    auto c = gsl_vector_alloc(roots);
    auto cov = gsl_matrix_alloc(roots, roots);

    gsl_multifit_linear_workspace* wspace = gsl_multifit_linear_alloc(xs.size(), roots);
    gsl_multifit_linear(X, Y, c, cov, &chisq, wspace);
    gsl_multifit_linear_free(wspace);

    std::vector<double> coeffs;
    for (int i {}; i < roots; ++i) {
        coeffs.push_back(gsl_vector_get(c, i));
    }

    return coeffs;
}