#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

#include <casacore/casa/Utilities/Compare.h>
#include <casacore/casa/Utilities/CountedPtr.h>
#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableIter.h>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "logger.h"
#include "outputtypes.h"

class DataTable {
public:
    struct Baseline {
        int a {};
        int b {};

        bool operator==(const Baseline& other) {
            return a == other.a && b == other.b;
        }

        bool operator!=(const Baseline& other) { return !(*this == other); }
    };

    struct RowMetadata {
        double time {};
        Baseline baseline {};
        double u {}, v {}, w{};
    };

    DataTable(
        const std::string& path,
        double maxduration,
        long long chanlow,
        long long chanhigh
    ) {
        casacore::MeasurementSet mset(path);

        // TODO: Allow selection of field ID.
        {
            auto field = mset.field();
            auto phasecenter = casacore::ArrayColumn<double>(
                field, "PHASE_DIR"
            ).get(0).tovector();
            m_phasecenter = {phasecenter[0], phasecenter[1]};
        }

        // Remove autocorrelations
        mset = mset(mset.col("ANTENNA1") != mset.col("ANTENNA2"));

        // Remove flagged rows
        mset = mset(!mset.col("FLAG_ROW"));

        // Retrieve frequency values
        {
            auto spw = mset.spectralWindow();
            casacore::ArrayColumn<double> chanCol(spw, "CHAN_FREQ");
            auto allfreqs = chanCol.get(0).tovector();

            // Ensure chanlow >= 0; and set chanhigh to default to all channels if <= 0
            chanlow = std::min(0ll, chanlow);
            if (chanhigh <= 0) chanhigh = allfreqs.size();

            for (long long chan {chanlow}; chan < chanhigh; ++chan) {
                m_freqs.push_back(allfreqs[chan]);
            }

            for (double freq : m_freqs) m_lambdas.push_back(Constants::c / freq);
        }

        // Set dimensions of table
        m_nrows = mset.nrow();
        m_nchans = chanhigh - chanlow;

        Logger::info(
            "Reading table with {} rows and {} channels (after filtering)...",
            m_nrows, m_nchans
        );

        // Allocate data arrays
        m_metadata.resize(m_nrows);
        m_weights.resize(m_nrows * m_nchans);
        m_data.resize(m_nrows * m_nchans);

        // Determine initial time for use in TableIterator
        double initialtime = [&] {
            auto times = casacore::ScalarColumn<double>(mset, "TIME_CENTROID").getColumn();
            return *std::min_element(times.begin(), times.end());
        }();

        if (maxduration <= 0) maxduration = std::numeric_limits<double>::max();

        // We iterate through the measurement set in [time block] x [baselines]
        // Each time block is <= maxduration. This order is essential for
        // efficient partitioning.
        casacore::Block<casacore::String> colnames(3);
        colnames[0] = "TIME_CENTROID";
        colnames[1] = "ANTENNA1";
        colnames[2] = "ANTENNA2";

        casacore::Block<std::shared_ptr<casacore::BaseCompare>> cmpobjs(3);
        auto interval = std::make_shared<casacore::CompareIntervalReal<double>>(
            initialtime, maxduration
        );
        cmpobjs[0] = interval;
        cmpobjs[1] = std::shared_ptr<casacore::BaseCompare>();
        cmpobjs[2] = std::shared_ptr<casacore::BaseCompare>();

        casacore::Block<casacore::Int> orders(3);
        orders[0] = 0;
        orders[1] = 0;
        orders[2] = 0;

        size_t irow {};
        for (
            casacore::TableIterator iter(mset, colnames, cmpobjs, orders);
            !iter.pastEnd();
            iter.next()
        ) {
            auto subtbl = iter.table();
            size_t nsubrows = subtbl.nrow();

            // Skip processing of an empty table
            if (nsubrows == 0) continue;

            int ant1 = casacore::ScalarColumn<int>(subtbl, "ANTENNA1").get(0);
            int ant2 = casacore::ScalarColumn<int>(subtbl, "ANTENNA2").get(0);

            // Create metadata
            {
                auto timeCol = casacore::ScalarColumn<double>(
                    subtbl, "TIME_CENTROID"
                ).getColumn();

                auto uvwCol = casacore::ArrayColumn<double>(subtbl, "UVW").getColumn();

                auto timeIter = timeCol.begin();
                auto uvwIter = uvwCol.begin();
                for (size_t i {}; i < nsubrows; ++i) {
                    double u = *uvwIter; ++uvwIter;
                    double v = *uvwIter; ++uvwIter;
                    double w = *uvwIter; ++uvwIter;

                    m_metadata[irow + i] = RowMetadata{
                        .time = *timeIter,
                        .baseline = Baseline{ant1, ant2},
                        .u = u, .v = v, .w = w
                    };

                    ++timeIter;
                }
            }

            // Copy weight spectrum in full
            {
                auto weightSpectrumCol = casacore::ArrayColumn<float>(
                    subtbl, "WEIGHT_SPECTRUM"
                ).getColumn();

                std::copy(
                    weightSpectrumCol.begin(), weightSpectrumCol.end(),
                    reinterpret_cast<float*>(m_weights.data() + irow * m_nchans)
                );
            }

            // Copy across the data
            {
                auto dataCol = casacore::ArrayColumn<std::complex<float>>(
                    subtbl, "CORRECTED_DATA"
                ).getColumn();

                std::copy(
                    dataCol.begin(), dataCol.end(),
                    reinterpret_cast<std::complex<float>*>(m_data.data() + irow * m_nchans)
                );
            }

            // Then fold flag and NaNs from the data column into weights
            {
                auto flagCol = casacore::ArrayColumn<bool>(subtbl, "FLAG").getColumn();
                auto flagIter = flagCol.begin();

                for (size_t i {}; i < nsubrows; ++i) {
                    for (size_t j {}; j < m_nchans; ++j) {
                        auto& datum = m_data[(irow + i) * m_nchans + j];
                        auto& weight = m_weights[(irow + i) * m_nchans + j];

                        // Treat NaNs as flags
                        if (!datum.isfinite() || !weight.isfinite()) {
                            datum = {0, 0, 0, 0};
                            weight = {0, 0, 0, 0};
                        }

                        // Fold flag values into the weight column
                        weight.xx *= !(*flagIter); ++flagIter;
                        weight.xy *= !(*flagIter); ++flagIter;
                        weight.yx *= !(*flagIter); ++flagIter;
                        weight.yy *= !(*flagIter); ++flagIter;
                    }
                }
            }

            irow += nsubrows;
        }
    }

    size_t size() const { return m_nrows * m_nchans; }
    size_t nrows() const { return m_nrows; }
    size_t nchans() const { return m_nchans; }

    const RaDec phasecenter() const { return m_phasecenter; }
    void phasecenter(RaDec radec) { m_phasecenter = radec; }

    const std::vector<double>& lambdas() const { return m_lambdas; }

    std::vector<RowMetadata>& metadata() { return m_metadata; }
    const std::vector<RowMetadata>& metadata() const { return m_metadata; }

    RowMetadata metadata(size_t i) const { return m_metadata[i]; }

    HostSpan<RowMetadata, 1> metadata(std::array<size_t, 2> rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 1>(static_cast<long long>(rowend - rowstart)),
            m_metadata.data() + rowstart
        };
    }

    std::array<double, 3> uvw(size_t row, size_t chan) {
        RowMetadata& m = m_metadata[row];
        double lambda = m_lambdas[chan];

        return {m.u / lambda, m.v / lambda, m.w / lambda};
    }

    HostSpan<ComplexLinearData<float>, 2> data(const std::array<size_t, 2>& rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 2>(
                static_cast<long long>(rowend - rowstart),
                static_cast<long long>(m_nchans)
            ),
            m_data.data() + rowstart * m_nchans
        };
    }

    ComplexLinearData<float>& data(const size_t row, const size_t chan) {
        return m_data[row * m_nchans + chan];
    }

    HostSpan<LinearData<float>, 2> weights() {
        return {
            std::array<long long, 2>(
                static_cast<long long>(m_nrows),
                static_cast<long long>(m_nchans)
            ),
            m_weights.data()
        };
    }

    HostSpan<LinearData<float>, 2> weights(const std::array<size_t, 2>& rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 2>(
                static_cast<long long>(rowend - rowstart),
                static_cast<long long>(m_nchans)
            ),
            m_weights.data() + rowstart * m_nchans
        };
    }

    LinearData<float>& weights(const size_t row, const size_t chan) {
        return m_weights[row * m_nchans + chan];
    }

    size_t mem() {
        return (
            m_freqs.size() * sizeof(double) +
            m_metadata.size() * sizeof(RowMetadata) +
            m_weights.size() * sizeof(LinearData<float>) +
            m_data.size() * sizeof(ComplexLinearData<float>)
        );
    }

private:
    // TODO: Investigate storing baselines as std::unorded_map?
    size_t m_nrows {};
    size_t m_nchans {};
    RaDec m_phasecenter;
    std::vector<double> m_freqs;
    std::vector<double> m_lambdas;
    std::vector<RowMetadata> m_metadata;
    std::vector<LinearData<float>> m_weights;
    std::vector<ComplexLinearData<float>> m_data;
};