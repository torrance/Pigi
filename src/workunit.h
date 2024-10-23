#pragma once

#include "datatable.h"

struct WorkUnit {
    double time;
    double freq;
    DataTable::Baseline baseline;

    // The center of a workunit is Nx / 2, Ny / 2, using integer division.
    // This is to be consistent with where the center is in an FFT.

    // The corresponding location within master grid [px] of the central pixel.
    // Note this is respect to the bottom left corner of the master grid.
    long long upx, vpx;

    // The u, v, w [dimensionless] values of the workunit center.
    double u, v, w;

    // Data slice values into the corresponding DataTable
    size_t rowstart, rowend;
    size_t chanstart, chanend;

    size_t size() const {
        return (rowend - rowstart) * (chanend - chanstart);
    }
};