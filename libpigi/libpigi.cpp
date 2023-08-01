#include <fmt/format.h>

#include "degridder.cpp"
#include "dft.cpp"
#include "gridder.cpp"
#include "invert.cpp"
#include "memory.cpp"
#include "outputtypes.cpp"
#include "predict.cpp"
#include "taper.cpp"
#include "workunit.cpp"

extern "C" {
    // void gridder_lineardouble(
    //     DeviceSpan<ComplexLinearData<double>, 2>* grid,
    //     WorkUnit<double>* workunits_ptr,
    //     size_t workunits_n,
    //     DeviceSpan<double, 2>* subtaper
    // ) {
    //     HostSpan<WorkUnit<double>, 1> workunits({workunits_n,}, workunits_ptr);
    //     gridder(*grid, workunits, *subtaper);
    // }

    // void gridder_stokesIdouble(
    //     DeviceSpan<StokesI<double>, 2>* grid,
    //     WorkUnit<double>* workunits_ptr,
    //     size_t workunits_n,
    //     DeviceSpan<double, 2>* subtaper
    // ) {
    //     HostSpan<WorkUnit<double>, 1> workunits({workunits_n}, workunits_ptr);
    //     gridder(*grid, workunits, *subtaper);
    // }

    // void gridder_stokesIfloat(
    //     DeviceSpan<StokesI<float>, 2>* grid,
    //     WorkUnit<float>* workunits_ptr,
    //     size_t workunits_n,
    //     DeviceSpan<float, 2>* subtaper
    // ) {
    //     HostSpan<WorkUnit<float>, 1> workunits({workunits_n}, workunits_ptr);
    //     gridder(*grid, workunits, *subtaper);
    // }

    void invert_stokesIdouble(
        StokesI<double>* img_ptr,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        GridSpec* gridspec
    ) {
        HostSpan<WorkUnit<double>, 1> workunits {{workunits_n}, workunits_ptr};

        auto subgridspec = workunits[0].subgridspec;

        auto taper = kaiserbessel<double>(*gridspec);
        auto subtaper = kaiserbessel<double>(subgridspec);

        auto img = invert<StokesI, double>(
            workunits, *gridspec, taper, subtaper
        );
        memcpy(img_ptr, img.data(), img.size() * sizeof(StokesI<double>));
    }

    void invert_stokesIfloat(
        StokesI<float>* img_ptr,
        HostSpan<UVDatum<float>, 1>* uvdata,
        GridSpec* gridspec,
        GridSpec* subgridspec,
        int padding,
        int wstep,
        HostSpan<ComplexLinearData<float>, 2>* Aterms
    ) {
        fmt::println("There are {} UVdatum", uvdata->size());
        {
            auto uvdatum = uvdata->front();
            fmt::println(
                "{} {} {} {} {}", uvdatum.u, uvdatum.v, uvdatum.w,
                uvdatum.weights, uvdatum.data
            );
        }

        std::vector<WorkUnit<float, HostArray<UVDatum<float>, 1>>> workunits = partition(
            *uvdata, *gridspec, *subgridspec, padding, wstep, *Aterms
        );

        // fmt::println("C++ there are {} workunits", _workunits.size());
        // {
        //     auto uvdatum = _workunits.front().data.front();
        //     fmt::println(
        //         "{} {} {} {} {}", uvdatum.u, uvdatum.v, uvdatum.w,
        //         uvdatum.weights, uvdatum.data
        //     );
        // }

        // std::vector<WorkUnit<float>> workunits;
        // for (auto& workunit : _workunits) {
        //     workunits.push_back(WorkUnit<float> {
        //         workunit.u0px, workunit.v0px,
        //         workunit.u0, workunit.v0, workunit.w0,
        //         workunit.subgridspec, workunit.Aleft, workunit.Aright,
        //         workunit.data.asSpan()
        //     });
        // }

        // fmt::println("C++ there are {} workunits", workunits.size());
        // {
        //     fmt::println("Workunit u0 {} v0 {} w0 {}",
        //         workunits.front().u0, workunits.front().v0, workunits.front().w0);
        //     auto uvdatum = workunits.front().data.front();
        //     fmt::println(
        //         "{} {} {} {} {}", uvdatum.u, uvdatum.v, uvdatum.w,
        //         uvdatum.weights, uvdatum.data
        //     );

        //     int count {};
        //     for (auto& workunit : workunits) {
        //         count += workunit.data.size();
        //     }
        //     fmt::println("There are {} UVDatum after partitioning", count);
        // }

        auto taper = kaiserbessel<float>(*gridspec);
        auto subtaper = kaiserbessel<float>(*subgridspec);

        auto img = invert<StokesI, float, HostArray<UVDatum<float>, 1>>(
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
        GridSpec* gridspec
    ) {
        auto taper = kaiserbessel<float>(*gridspec);
        auto subtaper = kaiserbessel<float>(workunits->front().subgridspec);
        predict<StokesI<float>, float>(
            *workunits, *img, *gridspec, taper, subtaper
        );
    }

    void predict_stokesIdouble(
        HostSpan<WorkUnit<double>, 1>* workunits,
        HostSpan<StokesI<double>, 2>* img,
        GridSpec* gridspec
    ) {
        auto taper = kaiserbessel<double>(*gridspec);
        auto subtaper = kaiserbessel<double>(workunits->front().subgridspec);
        predict<StokesI<double>, double>(
            *workunits, *img, *gridspec, taper, subtaper
        );
    }
}