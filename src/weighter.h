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

#include "datatable.h"
#include "gridspec.h"
#include "workunit.h"

class Weighter {
public:
    virtual const LinearData<float>& operator()(
        const double u, const double v, LinearData<float>& weight
    ) const = 0;
    virtual ~Weighter() = default;
};

void applyweights(const Weighter& weighter, DataTable& tbl) {
    LinearData<double> totalWeight {};  // The sum requires double precision

    // Apply weighting
    for (size_t irow {}; irow < tbl.nrows(); ++irow) {
        for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
            auto [u, v, _] = tbl.uvw(irow, ichan);
            totalWeight += static_cast<LinearData<double>>(
                weighter(u, v, tbl.weights(irow, ichan))
            );
        }
    }

    // Then normalize weight sum to unity
    for (auto& weight : tbl.weights()) {
        weight = static_cast<LinearData<float>>(
            static_cast<LinearData<double>>(weight) /= totalWeight
        );
    }
}

class Natural : public Weighter {
public:
    Natural() = delete;

    Natural(DataTable&, GridSpec gridspec) : gridspec(gridspec) {}

    inline const LinearData<float>& operator()(
        const double u, const double v, LinearData<float>& weight
    ) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(u, v);
        if (!(0 <= upx && upx < gridspec.Nx && 0 <= vpx && vpx < gridspec.Ny)) {
            weight = {0, 0, 0, 0};
        }
        return weight;
    }

private:
    GridSpec gridspec;
};

class Uniform : public Weighter {
public:
    Uniform() = delete;

    Uniform(DataTable& tbl, GridSpec gridspec)
        : gridspec(gridspec), griddedWeights{gridspec.Nx, gridspec.Ny} {

        // Sum weights for each grid cell
        for (size_t irow {}; irow < tbl.nrows(); ++irow) {
            for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
                auto [u, v, w] = tbl.uvw(irow, ichan);
                auto [upx, vpx] = gridspec.UVtoGrid(u, v);
                long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };

                if (
                    0 <= upx_ll && upx_ll < gridspec.Nx &&
                    0 <= vpx_ll && vpx_ll < gridspec.Ny
                ) {
                    griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)]
                        += tbl.weights(irow, ichan);
                }
            }
        }

        // Invert weights with special handling for 0 division
        for (auto& weight : griddedWeights) {
            for (size_t i {}; i < 4; ++i) {
                weight[i] = weight[i] > 0 ? 1 / weight[i] : 0;
            }
        }
    }

    inline const LinearData<float>& operator()(
        const double u, const double v, LinearData<float>& weight
    ) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(u, v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            weight *= griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            weight *= 0;
        }

        return weight;
    }

private:
    GridSpec gridspec;
    HostArray<LinearData<float>, 2> griddedWeights;
};

class Briggs : public Weighter {
public:
    Briggs(DataTable& tbl, GridSpec gridspec, double robust)
        : gridspec(gridspec), griddedWeights{gridspec.Nx, gridspec.Ny} {

        // Sum weights for each grid cell
        for (size_t irow {}; irow < tbl.nrows(); ++irow) {
            for (size_t ichan {}; ichan < tbl.nchans(); ++ichan) {
                auto [u, v, w] = tbl.uvw(irow, ichan);
                auto [upx, vpx] = gridspec.UVtoGrid(u, v);
                long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };

                if (
                    0 <= upx_ll && upx_ll < gridspec.Nx &&
                    0 <= vpx_ll && vpx_ll < gridspec.Ny
                ) {
                    griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)]
                        += tbl.weights(irow, ichan);
                }
            }
        }

        // Calculate f2
        LinearData<float> f2 { 1, 1, 1, 1};
        f2 *= std::pow(5 * std::pow(10, -robust), 2);

        LinearData<float> sumWk2, sumwi;
        for (auto weight : griddedWeights) {
            // Note: weight is a copy
            sumwi += weight;
            sumWk2 += (weight *= weight);
        }

        f2 /= (sumWk2 /= sumwi);

        // Apply f2 to inverse gridded weights and calculate norm as we go
        for (auto& weight: griddedWeights) {
            for (size_t i {}; i < 4; ++i) {
                float briggsWeight { 1 / (1 + weight[i] * f2[i]) };
                weight[i] = briggsWeight;
            }
        }
    }

    inline const LinearData<float>& operator()(
        const double u, const double v, LinearData<float>& weight
    ) const override {
        auto [upx, vpx] = gridspec.UVtoGrid(u, v);
        long long upx_ll { std::llround(upx) }, vpx_ll { std::llround(vpx) };
        if (
            0 <= upx_ll && upx_ll < gridspec.Nx &&
            0 <= vpx_ll && vpx_ll < gridspec.Ny
        ) {
            weight *= griddedWeights[gridspec.gridToLinear(upx_ll, vpx_ll)];
        } else {
            weight *= 0;
        }

        return weight;
    }


private:
    GridSpec gridspec;
    HostArray<LinearData<float>, 2> griddedWeights;
};