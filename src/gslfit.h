#pragma once

#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>

class GSLFit {
public:
    using fn_t = int (*)(const gsl_vector*, void*, gsl_vector*);
    class Exception : public std::runtime_error {
    public:
        Exception(const char* msg) : std::runtime_error(msg) {};
        const char* what() const noexcept { return std::runtime_error::what(); }
    };

    GSLFit(
        const fn_t fn, const size_t nparams, const size_t ndata
    ) : fn(fn), nparams(nparams), ndata(ndata), params(nparams) {
        if (ndata < nparams) {
            throw Exception("GSLFit: ndata must be at least equal to nparams");
        }

        // Use non-linear trust fit method
        const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;

        // Allocate the workspace with n-sized space and params0.size() parameters to solve
        gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
        w = gsl_multifit_nlinear_alloc(
            T, &fdf_params, ndata, nparams
        );

        // If the allocation failed, w will be a null pointer
        if (!w) {
            throw Exception(
                "GSLFit: An error occurred allocating the GSL multifit nlinear workspace"
            );
        }
    }

    GSLFit(const GSLFit&) = delete;
    GSLFit(GSLFit&&) = delete;
    GSLFit& operator=(const GSLFit&) = delete;
    GSLFit& operator=(GSLFit&&) = delete;

    ~GSLFit() {
        gsl_multifit_nlinear_free(w);
    }

    std::vector<double>& fit(const std::vector<double>& params0, void* data) {
        if (params0.size() != nparams) {
            throw Exception("GSLFit: nparams must equal params0.size()");
        }

        gsl_vector_view gslparams0 = gsl_vector_view_array(
            const_cast<double*>(params0.data()), params0.size()
        );

        // Define the function to be minimised
        gsl_multifit_nlinear_fdf fdf;
        fdf.f = fn;
        fdf.df = NULL; // using default finite-difference Jacobian
        fdf.n = ndata;
        fdf.p = nparams; // 3 parameters to solve for
        fdf.params = data;

        // Initialize the workspace with function and initial params guess
        int status;
        status = gsl_multifit_nlinear_init(&gslparams0.vector, &fdf, w);
        if (status != GSL_SUCCESS) {
            throw Exception("GSLFit: An error occurred initializing the GSL workspace");
        }

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
            break;
        case GSL_EMAXITER:
            Logger::warning("GSLFit: Fit reached max iterations before convergence.");
            break;
        case GSL_ENOPROG:
            throw Exception(
                "GSLFit: Fit did not converge "
                "(reason: no new acceptable delta could be found)"
            );
        default:
            throw Exception ("GSLFit: unknown error code returned from fitting");
        }

        // Copy fitted parameters and return
        for (size_t i {}; i < nparams; ++i) {
            params[i] = gsl_vector_get(w->x, i);
        }
        return params;
    }

private:
    fn_t fn;
    size_t nparams;
    size_t ndata; // number of x values
    gsl_multifit_nlinear_workspace* w;
    std::vector<double> params;
};