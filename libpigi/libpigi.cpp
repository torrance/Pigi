#include "array.cpp"
#include "gridder.cpp"
#include "invert.cpp"
#include "outputtypes.cpp"

extern "C" {
    void gridder_lineardouble(
        SpanMatrix<LinearData<double>>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<double>* subtaper
    ) {
        SpanVector<WorkUnit<double>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIdouble(
        SpanMatrix<StokesI<double>>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<double>* subtaper
    ) {
        SpanVector<WorkUnit<double>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIfloat(
        SpanMatrix<StokesI<float>>* grid,
        WorkUnit<float>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<float>* subtaper
    ) {
        SpanVector<WorkUnit<float>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void invert_stokesIdouble(
        StokesI<double>* img_ptr,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        GridSpec* gridspec,
        double* taper_ptr,
        double* subtaper_ptr
    ) {
        SpanVector<WorkUnit<double>> workunits {workunits_ptr, {workunits_n}};

        auto subgridspec = workunits[0].subgridspec;

        SpanMatrix<double> taper {
            taper_ptr, {(size_t) gridspec->Nx, (size_t) gridspec->Ny}
        };
        SpanMatrix<double> subtaper {
            subtaper_ptr, {(size_t) subgridspec.Nx, (size_t) subgridspec.Ny}
        };

        auto img = invert<StokesI, double>(
            workunits, *gridspec, taper, subtaper
        );
        memcpy(img_ptr, img.data(), img.size() * sizeof(StokesI<double>));
    }

    void invert_stokesIfloat(
        StokesI<float>* img_ptr,
        WorkUnit<float>* workunits_ptr,
        size_t workunits_n,
        GridSpec* gridspec,
        float* taper_ptr,
        float* subtaper_ptr
    ) {
        SpanVector<WorkUnit<float>> workunits {workunits_ptr, {workunits_n}};

        auto subgridspec = workunits[0].subgridspec;

        SpanMatrix<float> taper {
            taper_ptr, {(size_t) gridspec->Nx, (size_t) gridspec->Ny}
        };
        SpanMatrix<float> subtaper {
            subtaper_ptr, {(size_t) subgridspec.Nx, (size_t) subgridspec.Ny}
        };

        auto img = invert<StokesI, float>(
            workunits, *gridspec, taper, subtaper
        );
        memcpy(img_ptr, img.data(), img.size() * sizeof(StokesI<float>));
    }
}