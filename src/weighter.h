/**
 * # Image Weighting
 *
 * We use the image weighting descriptions from https://casa.nrao.edu/docs/casaref/imager.weight.html
 * which defines the weight of the i'th data point, w_i as:
 *
 * ## Natural
 *
 *    w_i = ω_i = 1 / σ_i^2
 *    where σ_i is the estimated noise of the visisbility sample
 *
 * In practice, we take the ω_i as the row weight x the weight spectrum value from the
 * measurement set, and assume this has been set correctly by the observatory.
 *
 * ## Uniform
 *
 *     w_i = ω_i / W_k
 *     where W_k = ∑ ω_i within the k'th UV cell
 *
 * ## Briggs
 *
 *     w_i = ____ω_i____
 *           1 + W_k f^2
 *
 *     where f^2 =  ___(5_*_10^-R)^2___
 *                  ∑_k W_k^2 / ∑_i ω_i
 */

#include <vector>

#include "gridspec.h"
#include "uvdatum.h"
#include "workunit.h"

template <typename T>
class Weighter {
public:
    virtual const LinearData<T>& operator()(UVDatum<T>&) const;
    virtual ~Weighter() = default;
};

template <typename T, typename R>
LinearData<T> applyWeights(const Weighter<T>& weighter, R& uvdata) {
    LinearData<T> totalWeight {};

    for (UVDatum<T>& uvdatum : uvdata) {
        totalWeight += weighter(uvdatum);
    }

    return totalWeight;
}

template <typename T>
LinearData<T> applyWeights(const Weighter<T>& weighter, std::vector<WorkUnit<T>>& workunits) {
    LinearData<T> totalWeight {};

    for (WorkUnit<T>& workunit : workunits) {
        for (UVDatum<T>& uvdatum : workunit.data) {
            totalWeight += weighter(uvdatum);
        }
    }

    return totalWeight;
}

template <typename T>
class Natural : public Weighter<T> {
public:
    Natural() = delete;

    template <typename R>
    Natural(R&& uvdata, GridSpec gridspec) {}

    inline const LinearData<T>& operator()(UVDatum<T>& uvdatum) const override {
        return uvdatum.weights;
    }
};

template <typename T>
class Uniform : public Weighter<T> {
public:
    Uniform() = delete;

    template <typename R>
    Uniform(R&& uvdata, GridSpec gridspec)
        : gridspec(gridspec), griddedWeights{gridspec.Nx, gridspec.Ny} {

        // Sum weights for each grid cell
        for (const UVDatum<T>& uvdatum : uvdata) {
            auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
            long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };

            if (
                0 <= upx_ll && upx_ll < gridspec.Nx &&
                0 <= vpx_ll && vpx_ll < gridspec.Ny
            ) {
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)] += uvdatum.weights;
            }
        }

        // Invert weights with special handling for 0 division
        for (auto& weight : griddedWeights) {
            for (size_t i {}; i < 4; ++i) {
                weight[i] = weight[i] > 0 ? 1 / weight[i] : 0;
            }
        }
    }

    inline const LinearData<T>& operator()(UVDatum<T>& uvdatum) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            uvdatum.weights *=
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            uvdatum.weights *= 0;
        }

        return uvdatum.weights;
    }

private:
    GridSpec gridspec;
    HostArray<LinearData<T>, 2> griddedWeights;
};

template <typename T>
class Briggs : public Weighter<T> {
public:
    template <typename R>
    Briggs(R&& uvdata, GridSpec gridspec, double robust)
        : robust(robust), gridspec(gridspec), griddedWeights{gridspec.Nx, gridspec.Ny} {

        // Sum weights for each grid cell
        for (const UVDatum<T>& uvdatum : uvdata) {
            auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
            long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };

            if (
                0 <= upx_ll && upx_ll < gridspec.Nx &&
                0 <= vpx_ll && vpx_ll < gridspec.Ny
            ) {
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)] += uvdatum.weights;
            }
        }

        // Calculate f2
        LinearData<T> f2 { 1, 1, 1, 1};
        f2 *= std::pow(5 * std::pow(10, -robust), 2);

        LinearData<T> sumWk2, sumwi;
        for (auto weight : griddedWeights) {
            // Note: weight is a copy
            sumwi += weight;
            sumWk2 += (weight *= weight);
        }

        f2 /= (sumWk2 /= sumwi);

        // Apply f2 to inverse gridded weights and calculate norm as we go
        for (auto& weight: griddedWeights) {
            for (size_t i {}; i < 4; ++i) {
                T briggsWeight { 1 / (1 + weight[i] * f2[i]) };
                weight[i] = briggsWeight;
            }
        }
    }

    inline const LinearData<T>& operator()(UVDatum<T>& uvdatum) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            uvdatum.weights *=
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            uvdatum.weights *= 0;
        }

        return uvdatum.weights;
    }


private:
    double robust {};
    GridSpec gridspec;
    HostArray<LinearData<T>, 2> griddedWeights;
};