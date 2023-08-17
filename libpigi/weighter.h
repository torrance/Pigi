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
    virtual void operator()(UVDatum<T>&) const;
};

template <typename T, typename R>
void applyWeights(const Weighter<T>& weighter, R& uvdata) {
    for (UVDatum<T>& uvdatum : uvdata) {
        weighter(uvdatum);
    }
}

template <typename T>
void applyWeights(const Weighter<T>& weighter, std::vector<WorkUnit<T>>& workunits) {
    for (WorkUnit<T>& workunit : workunits) {
        for (UVDatum<T>& uvdatum : workunit.data) {
            weighter(uvdatum);
        }
    }
}

template <typename T>
class Natural : public Weighter<T> {
public:
    Natural() = delete;

    template <typename R>
    Natural(const R& uvdata, GridSpec gridspec) {
        for (const UVDatum<T>& uvdatum : uvdata) {
            // Check if (u, v) lie on the grid and sum
            auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
            long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
            if (
                0 <= upx_ll && upx_ll < gridspec.Nx &&
                0 <= vpx_ll && vpx_ll < gridspec.Ny
            ) {
                norm += uvdatum.weights;
            }
        }

        // Invert the norm with special handling for 0 weights
        norm.xx = norm.xx > 0 ? 1 / norm.xx : 0;
        norm.yx = norm.yx > 0 ? 1 / norm.yx : 0;
        norm.xy = norm.xy > 0 ? 1 / norm.xy : 0;
        norm.yy = norm.yy > 0 ? 1 / norm.yy : 0;
    }

    inline void operator()(UVDatum<T>& uvdatum) override {
        uvdatum.data *= norm;
    }

private:
    LinearData<T> norm {};
};

template <typename T>
class Uniform : public Weighter<T> {
public:
    Uniform() = delete;

    template <typename R>
    Uniform(const R& uvdata, GridSpec gridspec)
        : gridspec(gridspec), griddedWeights({gridspec.Nx, gridspec.Ny}) {

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
        // and calculate norm as sum of all non-zero grid cells
        for (auto& weight : griddedWeights) {
            for (size_t i {}; i < 4; ++i) {
                weight[i] = weight[i] > 0 ? 1 / weight[i] : 0;
                norm[i] += weight[i] == 0 ? 0 : 1;
            }
        }

        // Finally, invert the norm with special handling for 0 division
        norm.xx = norm.xx > 0 ? 1 / norm.xx : 0;
        norm.yx = norm.yx > 0 ? 1 / norm.yx : 0;
        norm.xy = norm.xy > 0 ? 1 / norm.xy : 0;
        norm.yy = norm.yy > 0 ? 1 / norm.yy : 0;
     }

    inline void operator()(UVDatum<T>& uvdatum) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            uvdatum.weights *= norm;
            uvdatum.weights *=
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            uvdatum.weights *= 0;
        }
    }

private:
    GridSpec gridspec;
    LinearData<T> norm;
    HostArray<LinearData<T>, 2> griddedWeights;
};

template <typename T>
class Briggs : public Weighter<T> {
public:
    template <typename R>
    Briggs(R& uvdata, GridSpec gridspec, double robust)
        : robust(robust), gridspec(gridspec), griddedWeights({gridspec.Nx, gridspec.Ny}) {

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
                norm[i] += briggsWeight * weight[i];
                weight[i] = briggsWeight;
            }
        }

        // Finally, invert the norm with special handling for 0 division
        norm.xx = norm.xx > 0 ? 1 / norm.xx : 0;
        norm.yx = norm.yx > 0 ? 1 / norm.yx : 0;
        norm.xy = norm.xy > 0 ? 1 / norm.xy : 0;
        norm.yy = norm.yy > 0 ? 1 / norm.yy : 0;
    }

    inline void operator()(UVDatum<T>& uvdatum) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(uvdatum.u, uvdatum.v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            uvdatum.weights *= norm;
            uvdatum.weights *=
                griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            uvdatum.weights *= 0;
        }
    }


private:
    double robust {};
    GridSpec gridspec;
    LinearData<T> norm {};
    HostArray<LinearData<T>, 2> griddedWeights;
};