#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/tables/Tables.h>
#include <casacore/tables/Tables/TableIter.h>
#include <fmt/format.h>
#include <thrust/complex.h>

#include "constants.h"
#include "coordinates.h"
#include "logger.h"
#include "outputtypes.h"
#include "timer.h"

class DataTable {
public:
    enum class DataColumn {automatic, data, corrected, model};

    struct Config {
        long long chanlow {};
        long long chanhigh {};
        DataColumn datacolumn {DataColumn::automatic};
        std::optional<RaDec> phasecenter {std::nullopt};
        bool skipdata {false};
    };

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

    DataTable() = default;

    DataTable(
        const std::string& fname, const Config& config
    ) : DataTable({casacore::MeasurementSet(fname)}, config) {}

    DataTable(
        std::vector<casacore::MeasurementSet> msets, const Config& config
    ) {
        auto timer = Timer::get("dataload");

        if (msets.empty()) return;

        m_datacolumn = config.datacolumn;

        for (size_t i {}; auto& mset : msets) {
            std::string fname = mset.tableName();

            // Remove autocorrelations
            mset = mset(mset.col("ANTENNA1") != mset.col("ANTENNA2"));

            // Remove flagged rows
            mset = mset(!mset.col("FLAG_ROW"));

            // Compute total rows across all msets
            m_nrows += mset.nrow();

            // Ensure just one spectral window
            auto spw = mset.spectralWindow();
            if (spw.nrow() != 1) throw std::runtime_error(fmt::format(
                "Found {} spectral windows in {}; expected 1", spw.nrow(), fname
            ));

            auto freqs = casacore::MSSpWindowColumns(spw)
                .chanFreq().get(0).tovector();

            // Initialize freqs if this is the first mset
            if (i == 0) m_freqs = freqs;

            // Ensure each freqs is identical across all msets
            if (m_freqs != freqs) throw std::runtime_error(
                "Each measurement set must have the same spectral window"
            );

            // Ensure channels are in range and set chanhigh to default value if == 0
            m_chanlow = config.chanlow;
            m_chanhigh = std::min<long long>(freqs.size(), config.chanhigh);
            if (m_chanhigh == 0) m_chanhigh = freqs.size();
            if (m_chanlow < 0) throw std::runtime_error(fmt::format(
                "Channel low must be > 0; got {}", m_chanlow
            ));
            if (!(m_chanlow < m_chanhigh)) throw std::runtime_error(fmt::format(
                "Channel low must be less than channel high (got low {}, high {})",
                m_chanlow, m_chanhigh
            ));

            // Ensure just one field
            auto field = mset.field();
            if (field.nrow() != 1) throw std::runtime_error(fmt::format(
                "Found {} fields in {}; expected 1", field.nrow(), fname
            ));

            // Set phase center if unset using first mset's phasecenter
            if (i == 0 && !config.phasecenter) {
                auto phasecenter = casacore::MSFieldColumns(field)
                    .phaseDir().get(0).tovector();

                m_phasecenter = {phasecenter.at(0), phasecenter.at(1)};
            }

            // Ensure data column exists; and all msets have the same datacolumn available
            // Also set datacolumn if DataColumn::automatic is set
            auto hascol = [] (const auto& mset, const auto& colname) -> bool {
                for (const auto& othercolname : mset.tableDesc().columnNames()) {
                    if (casacore::String(colname) == othercolname) return true;
                }
                return false;
            };

            switch (m_datacolumn) {
            case DataColumn::automatic:
                if (hascol(mset, "CORRECTED_DATA")) {
                    Logger::info("Automatically selecting CORRECTED_DATA column");
                    m_datacolumn = DataColumn::corrected;
                } else if (hascol(mset, "DATA")) {
                    m_datacolumn = DataColumn::data;
                    Logger::info("No CORRECTED_DATA column found; automatically using DATA instead");
                } else {
                    throw std::runtime_error(fmt::format(
                        "datacolumn=auto but neither DATA nor CORRECTED_DATA exist in {}",
                        mset.tableName()
                    ));
                }
                break;
            case DataColumn::data:
                if (!hascol(mset, "DATA")) throw std::runtime_error(fmt::format(
                    "DATA column not found in {}", mset.tableName()
                ));
                break;
            case DataColumn::corrected:
                if (!hascol(mset, "CORRECTED_DATA")) throw std::runtime_error(fmt::format(
                    "CORRECTED_DATA column not found in {}", mset.tableName()
                ));
                break;
            case DataColumn::model:
                if (!hascol(mset, "MODEL_DATA")) throw std::runtime_error(fmt::format(
                    "MODEL_DATA column not found in {}", mset.tableName()
                ));
                break;
            default:
                throw std::runtime_error("Invalid MeasurementSet::DataColumn value");
            }
        }

        // Set phasecenter if it has been explicitly provided
        if (config.phasecenter) m_phasecenter = *(config.phasecenter);

        // Set dimensions of table
        m_nchans = m_chanhigh - m_chanlow;

        // Truncate freqs just to channel range, and construct lambdas
        m_freqs.erase(m_freqs.begin(), m_freqs.begin() + m_chanlow);
        m_freqs.resize(m_nchans);

        // Create m_lambdas
        for (double freq : m_freqs) m_lambdas.push_back(Constants::c / freq);

        // If skipdata is true, we only get metadata on the msets and perform validation
        if (config.skipdata) {
            m_nrows = 0;
            return;
        }

        Logger::info(
            "Reading table with {} rows and {} channels [{}-{})...",
            m_nrows, m_nchans, m_chanlow, m_chanhigh
        );

        // Allocate data arrays
        m_metadata.resize(m_nrows);
        m_weights = HostArray<LinearData<float>, 2>(m_nchans, m_nrows);
        m_data = HostArray<ComplexLinearData<float>, 2>(m_nchans, m_nrows);

        // We iterate through the measurement by baseline
        casacore::Block<casacore::String> colnames(2);
        colnames[0] = "ANTENNA1";
        colnames[1] = "ANTENNA2";

        for (size_t irow {}; auto& mset : msets) {
            for (
                casacore::TableIterator iter(mset, colnames);
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
                        double v = -*uvwIter; ++uvwIter;  // Invert V [why??]
                        double w = *uvwIter; ++uvwIter;

                        m_metadata[irow + i] = RowMetadata{
                            .time = *timeIter / 86400.,  // Convert from mjd [seconds] to [days]
                            .baseline = Baseline{ant1, ant2},
                            .u = u, .v = v, .w = w
                        };

                        ++timeIter;
                    }
                }

                // Create slicers used to filter channels
                casacore::Slicer rowSlice {
                    casacore::IPosition {0},
                    casacore::IPosition {static_cast<long>(nsubrows)},
                    casacore::Slicer::endIsLength
                };

                casacore::Slicer arraySlice {
                    casacore::IPosition {0, m_chanlow},
                    casacore::IPosition {4, static_cast<long>(m_nchans)},
                    casacore::Slicer::endIsLength
                };

                // Copy weight spectrum in full
                {
                    auto weightSpectrumCol = casacore::ArrayColumn<float>(
                        subtbl, "WEIGHT_SPECTRUM"
                    ).getColumnRange(rowSlice, arraySlice);

                    std::copy(
                        weightSpectrumCol.begin(), weightSpectrumCol.end(),
                        reinterpret_cast<float*>(m_weights.data() + irow * m_nchans)
                    );
                }

                // Copy across the data
                {
                    casacore::Array<std::complex<float>> dataCol;

                    switch (m_datacolumn) {
                    case DataColumn::data:
                        dataCol = casacore::ArrayColumn<std::complex<float>>(
                            subtbl, "DATA"
                        ).getColumnRange(rowSlice, arraySlice);
                        break;
                    case DataColumn::corrected:
                        dataCol = casacore::ArrayColumn<std::complex<float>>(
                            subtbl, "CORRECTED_DATA"
                        ).getColumnRange(rowSlice, arraySlice);
                        break;
                    case DataColumn::model:
                        dataCol = casacore::ArrayColumn<std::complex<float>>(
                            subtbl, "MODEL_DATA"
                        ).getColumnRange(rowSlice, arraySlice);
                        break;
                    case DataColumn::automatic:
                    default:
                        throw std::runtime_error("m_datacolumn should not still be set to auto");
                    }

                    std::copy(
                        dataCol.begin(), dataCol.end(),
                        reinterpret_cast<std::complex<float>*>(m_data.data() + irow * m_nchans)
                    );
                }

                // Then fold flag and NaNs from the data column into weights
                {
                    auto flagCol = casacore::ArrayColumn<bool>(
                        subtbl, "FLAG"
                    ).getColumnRange(rowSlice, arraySlice);
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

                // Ensure all phase centers are set to m_phasecenter
                // We have to do this per mset, rather than once at the end,
                // since each mset might be initialized with a different phase center.
                {
                    auto timer = Timer::get("dataload::phaserotate");

                    RaDec subphasecenter = [&] () -> RaDec {
                        auto phasecenter = casacore::MSFieldColumns(
                            casacore::MeasurementSet(subtbl).field()
                        ).phaseDir().get(0).tovector();
                        return {phasecenter.at(0), phasecenter.at(1)};
                    }();

                    for (size_t i {}; i < nsubrows; ++i) {
                        phaserotaterow(subphasecenter, m_phasecenter, irow + i);
                    }
                }

                irow += nsubrows;
            }
        }

        Logger::verbose("Forcing visibility data to have positive w values...");
        forcewpositive();
    }

    size_t size() const { return m_nrows * m_nchans; }
    size_t nrows() const { return m_nrows; }
    size_t nchans() const { return m_nchans; }

    long long chanlow() const { return m_chanlow; }
    long long chanhigh() const { return m_chanhigh; }

    DataTable& forcewpositive() {
        // All visibilities with negative w values can be transformed to have postive
        // w values by swapping the order of the correlation. i.e. antenna 1 x 2 => 2 x 1.
        // This gives V(u, v, w)^H = V(-u, -v, -w)
        for (size_t irow {}; irow < m_nrows; ++irow) {
            auto& m = m_metadata[irow];
            if (m.w < 0) {
                // Swap antenna order
                std::swap(m.baseline.a, m.baseline.b);

                // Set all u,v,w as inverse
                m.u *= -1;
                m.v *= -1;
                m.w *= -1;

                // Finally apply adjoint to visibility datum
                auto datarow = data({irow, irow});
                auto weightsrow = weights({irow, irow});
                for (size_t ichan {}; ichan < m_nchans; ++ichan) {
                    datarow[ichan] = datarow[ichan].adjoint();
                    // We also need to transpose the weights.
                    // Note that for for real values, trans() = adjoint()
                    weightsrow[ichan] = weightsrow[ichan].adjoint();
                }
            }
        }

        return *this;
    }

    RaDec phasecenter() const { return m_phasecenter; }

    void phasecenter(RaDec phasecenter) {
        for (size_t irow {}; irow < m_nrows; ++irow) {
            phaserotaterow(m_phasecenter, phasecenter, irow);
        }
        m_phasecenter = phasecenter;
    }

    DataColumn datacolumn() const { return m_datacolumn; }

    double midfreq() const {
        return std::accumulate(m_freqs.begin(), m_freqs.end(), 0.) / m_freqs.size();
    }

    double midtime() const {
        auto [low, high] = std::minmax_element(
            m_metadata.begin(), m_metadata.end(), [] (auto lhs, auto rhs) {
                return lhs.time < rhs.time;
            }
        );
        return (low->time + high->time) / 2;
    }

    const std::vector<double>& freqs() const { return m_freqs; }

    HostSpan<double, 1> freqs(std::array<size_t, 2> chanslice) {
        auto& [chanstart, chanend] = chanslice;
        return {
            std::array<long long, 1>{static_cast<long long>(chanend - chanstart)},
            m_freqs.data() + chanstart
        };
    }

    using FreqRange = std::array<double, 2>;
    FreqRange freqrange() const {
        return {m_freqs.front(), m_freqs.back()};
    }

    const std::vector<double>& lambdas() const { return m_lambdas; }

    std::vector<RowMetadata>& metadata() { return m_metadata; }
    const std::vector<RowMetadata>& metadata() const { return m_metadata; }

    RowMetadata metadata(size_t i) const { return m_metadata[i]; }

    HostSpan<RowMetadata, 1> metadata(std::array<size_t, 2> rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 1>{static_cast<long long>(rowend - rowstart)},
            m_metadata.data() + rowstart
        };
    }

    std::array<double, 3> uvw(size_t row, size_t chan) const {
        const RowMetadata& m = m_metadata[row];
        double lambda = m_lambdas[chan];

        return {m.u / lambda, m.v / lambda, m.w / lambda};
    }

    HostSpan<ComplexLinearData<float>, 2> data() {
        return {
            std::array<long long, 2>{
                static_cast<long long>(m_nchans),
                static_cast<long long>(m_nrows)
	    },
            m_data.data()
        };
    }

    HostSpan<ComplexLinearData<float>, 2> data(const std::array<size_t, 2>& rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 2>{
                static_cast<long long>(m_nchans),
                static_cast<long long>(rowend - rowstart)
	    },
            m_data.data() + rowstart * m_nchans
        };
    }

    ComplexLinearData<float>& data(const size_t row, const size_t chan) {
        return m_data[row * m_nchans + chan];
    }

    HostSpan<LinearData<float>, 2> weights() {
        return {
            std::array<long long, 2>{
                static_cast<long long>(m_nchans),
                static_cast<long long>(m_nrows)
	    },
            m_weights.data()
        };
    }

    HostSpan<LinearData<float>, 2> weights(const std::array<size_t, 2>& rowslice) {
        auto& [rowstart, rowend] = rowslice;
        return {
            std::array<long long, 2>{
                static_cast<long long>(m_nchans),
                static_cast<long long>(rowend - rowstart)
	    },
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
    DataColumn m_datacolumn {DataColumn::automatic};
    long long m_chanlow {};
    long long m_chanhigh {};
    size_t m_nrows {};
    size_t m_nchans {};
    RaDec m_phasecenter;
    std::vector<double> m_freqs;
    std::vector<double> m_lambdas;
    std::vector<RowMetadata> m_metadata;
    HostArray<LinearData<float>, 2> m_weights;
    HostArray<ComplexLinearData<float>, 2> m_data;

    void phaserotaterow(RaDec from, RaDec to, size_t irow) {
        if (to == from) return;

        const double cos_deltara = std::cos(to.ra - from.ra);
        const double sin_deltara = std::sin(to.ra - from.ra);
        const double sin_decfrom = std::sin(from.dec);
        const double cos_decfrom = std::cos(from.dec);
        const double sin_decto = std::sin(to.dec);
        const double cos_decto = std::cos(to.dec);

        RowMetadata& m = m_metadata[irow];

        double u = m.u;
        double v = m.v;
        double w = m.w;

        const double uprime = (
            + u * cos_deltara
            - v * sin_decfrom * sin_deltara
            - w * cos_decfrom * sin_deltara
        );
        const double vprime = (
            + u * sin_decto * sin_deltara
            + v * (sin_decfrom * sin_decto * cos_deltara + cos_decfrom * cos_decto)
            - w * (sin_decfrom * cos_decto - cos_decfrom * sin_decto * cos_deltara)
        );
        const double wprime = (
            + u * cos_decto * sin_deltara
            - v * (cos_decfrom * sin_decto - sin_decfrom * cos_decto * cos_deltara)
            + w * (sin_decfrom * sin_decto + cos_decfrom * cos_decto * cos_deltara)
        );

        m.u = uprime;
        m.v = vprime;
        m.w = wprime;

        // Add in geometric delay to data
        for (size_t ichan {}; const double lambda : m_lambdas) {
            this->data(irow, ichan++) *= cispi(-2 * (wprime - w) / lambda);
        }
    }
};