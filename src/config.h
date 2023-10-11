#pragma once

#include <string>

#include "gridspec.h"
#include "mset.h"

struct Config {
    int precision;
    GridSpec gridspec;
    GridSpec gridspecPadded;
    GridSpec subgridspec;
    int chanlow;
    int chanhigh;
    int channelsOut;
    double maxDuration;
    int wstep;
    int kernelpadding;
    std::map<double, std::vector<MeasurementSet>> msets;
    std::string weight;
    float robust;
    float padding;

    // Clean parameters
    float majorgain;
    float minorgain;
    float cleanThreshold;
    float autoThreshold;
    size_t nMajor;
    size_t nMinor;
};