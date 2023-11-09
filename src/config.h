#pragma once

#include <string>

#include "gridspec.h"
#include "mset.h"

struct Config {
    int precision;

    // Measurement set selection
    int chanlow;
    int chanhigh;
    int channelsOut;
    double maxDuration;
    std::map<int, std::vector<MeasurementSet>> msets;

    // Data weigting
    std::string weight;
    float robust;

    GridConfig gridconf;

    // Clean parameters
    float majorgain;
    float minorgain;
    float cleanThreshold;
    float autoThreshold;
    size_t nMajor;
    size_t nMinor;
};