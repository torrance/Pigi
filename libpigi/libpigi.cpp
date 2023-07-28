#include "degridder.cpp"
#include "dft.cpp"
#include "gridder.cpp"
#include "invert.cpp"
#include "memory.cpp"
#include "outputtypes.cpp"
#include "predict.cpp"

extern "C" {
    void gridder_lineardouble(
        DeviceSpan<ComplexLinearData<double>, 2>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        DeviceSpan<double, 2>* subtaper
    ) {
        HostSpan<WorkUnit<double>, 1> workunits({workunits_n,}, workunits_ptr);
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIdouble(
        DeviceSpan<StokesI<double>, 2>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        DeviceSpan<double, 2>* subtaper
    ) {
        HostSpan<WorkUnit<double>, 1> workunits({workunits_n}, workunits_ptr);
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIfloat(
        DeviceSpan<StokesI<float>, 2>* grid,
        WorkUnit<float>* workunits_ptr,
        size_t workunits_n,
        DeviceSpan<float, 2>* subtaper
    ) {
        HostSpan<WorkUnit<float>, 1> workunits({workunits_n}, workunits_ptr);
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
        HostSpan<WorkUnit<double>, 1> workunits {{workunits_n}, workunits_ptr};

        auto subgridspec = workunits[0].subgridspec;

        HostSpan<double, 2> taper {
            {gridspec->Nx, gridspec->Ny}, taper_ptr
        };
        HostSpan<double, 2> subtaper {
            {subgridspec.Nx, subgridspec.Ny}, subtaper_ptr
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
        HostSpan<WorkUnit<float>, 1> workunits {{workunits_n}, workunits_ptr};

        auto subgridspec = workunits[0].subgridspec;

        HostSpan<float, 2> taper {
            {gridspec->Nx, gridspec->Ny}, taper_ptr
        };
        HostSpan<float, 2> subtaper {
            {subgridspec.Nx, subgridspec.Ny}, subtaper_ptr
        };

        auto img = invert<StokesI, float>(
            workunits, *gridspec, taper, subtaper
        );
        memcpy(img_ptr, img.data(), img.size() * sizeof(StokesI<float>));
    }

    void idft_lineardouble(
        DeviceSpan<ComplexLinearData<double>, 2>* img,
        DeviceSpan<UVDatum<double>, 1>* uvdata,
        GridSpec* gridspec,
        double normfactor
    ) {
        idft<ComplexLinearData<double>, double>(
            *img, *uvdata, *gridspec, normfactor
        );
    }

    void idft_linearfloat(
        DeviceSpan<ComplexLinearData<float>, 2>* img,
        DeviceSpan<UVDatum<float>, 1>* uvdata,
        GridSpec* gridspec,
        float normfactor
    ) {
        idft<ComplexLinearData<float>, float>(
            *img, *uvdata, *gridspec, normfactor
        );
    }

    void predict_stokesIfloat(
        HostSpan<WorkUnit<float>, 1>* workunits,
        HostSpan<StokesI<float>, 2>* img,
        GridSpec* gridspec,
        HostSpan<float, 2>* taper,
        HostSpan<float, 2>* subtaper
    ) {
        predict<StokesI<float>, float>(
            *workunits, *img, *gridspec, *taper, *subtaper
        );
    }

    void predict_stokesIdouble(
        HostSpan<WorkUnit<double>, 1>* workunits,
        HostSpan<StokesI<double>, 2>* img,
        GridSpec* gridspec,
        HostSpan<double, 2>* taper,
        HostSpan<double, 2>* subtaper
    ) {
        predict<StokesI<double>, double>(
            *workunits, *img, *gridspec, *taper, *subtaper
        );
    }
}