struct MeasurementSet <: DataStore
    tbl::PyCall.PyObject
    timestart::Float64
    timestop::Float64
    chanstart::Int
    chanstop::Int
    nrows::Int
    freqs::Vector{Float64}
    lambdas::Vector{Float64}
    times::Vector{Float64}
    timesteps::Vector{Float64}
    # antennas::Dict{Int, Tuple{String, @NamedTuple{x::Float64, y::Float64, z::Float64}}}
    ants1::Vector{Int}
    ants2::Vector{Int}
    phasecenter::NamedTuple{(:ra, :dec), Tuple{Float64, Float64}}  # radians
end

"""
    MeasurementSet(path::String, <keyword arguments>)

The default constructor for a MeasurementSet.

# Arguments
- `timestart=0`, `timestop=0` An optional time range (start and end inclusive). The default
will select the entire time range. Time is given in Modified Julian Date (MJD) _in seconds_
(i.e. standard MJD * 24 * 60 * 60).
- `chanstart=0`, `chanstop=0` An optional channel range (start and end inclusive). The
default will select the entire spectral window. Channels are 1-indexed.
- `autocorrelations=false` A flag to enable the inclusion of autocorrelations.
- `flagrow=false` A flag to enable the inclusion of rows that have been flagged.

"""
function MeasurementSet(
    path::String;
    timestart=0,
    timestop=0,
    chanstart::Int=0,
    chanstop::Int=0,
    autocorrelations=false,
    flagrow=false
)
    PyCasaTable = PyCall.pyimport("casacore.tables")
    tbl = PyCasaTable.table(path, ack=false)

    # Apply additional filters
    conditions = String[]
    if timestart != 0 || timestop != 0
        push!(conditions, "TIME >= $(timestart) and TIME <= $(timestop)")
    end
    if !autocorrelations
        push!(conditions, "ANTENNA1 <> ANTENNA2")
    end
    if !flagrow
        push!(conditions, "not FLAG_ROW")
    end
    if length(conditions) != 0
        tbl = PyCasaTable.taql("select * from \$1 where " * join(conditions, " and "), tables=[tbl])
    end

    freqs = tbl.SPECTRAL_WINDOW.getcellslice("CHAN_FREQ", 0, chanstart - 1, chanstop - 1)
    lambdas = c ./ freqs
    ra0, dec0 = deg2rad.(tbl.FIELD.getcell("PHASE_DIR", 0)[1, :])

    # Get time information
    times = tbl.getcol("TIME")
    timesteps = unique(times)
    sort!(timesteps)

    ants1 = tbl.getcol("ANTENNA1")
    ants2 = tbl.getcol("ANTENNA2")

    return MeasurementSet(
        tbl,
        timestart,
        timestop,
        chanstart,
        chanstop,
        tbl.nrows(),
        freqs,
        lambdas,
        times,
        timesteps,
        ants1,
        ants2,
        (ra=ra0, dec=dec0),
    )
end

function read(mset::MeasurementSet; precision::Type=Float64, datacol=nothing)
    if datacol === nothing
        datacol = "CORRECTED_DATA" in mset.tbl.colnames() ? "CORRECTED_DATA" :  "DATA"
    end

    nbatch = 1000  # Rows of measurement set to read from disk at a time

    # Convert channel selection from 1-indexing to Python's 0-indexing
    chanstart = mset.chanstart - 1
    chanstop = mset.chanstop - 1

    ch = Channel{UVDatum{precision}}(10000) do ch
        for startrow in 1:nbatch:mset.nrows
            startrow -= 1  # Convert from 1 indexing to Python's 0-indexing

            uvw = PermutedDimsArray(PyCall.pycall(
                mset.tbl.getcol, PyCall.PyArray, "UVW", startrow=startrow, nrow=nbatch
            ), (2, 1))
            flagrow = PyCall.pycall(
                mset.tbl.getcol, PyCall.PyArray, "FLAG_ROW", startrow=startrow, nrow=nbatch
            )
            weight = PermutedDimsArray(PyCall.pycall(
                mset.tbl.getcol, PyCall.PyArray, "WEIGHT", startrow=startrow, nrow=nbatch
            ), (2, 1))
            flag = PermutedDimsArray(PyCall.pycall(
                mset.tbl.getcolslice, PyCall.PyArray, "FLAG",
                startrow=startrow, nrow=nbatch, blc=(chanstart, -1), trc=(chanstop, -1)
            ), (3, 2, 1))
            weightspectrum = PermutedDimsArray(PyCall.pycall(
                mset.tbl.getcolslice, PyCall.PyArray, "WEIGHT_SPECTRUM",
                startrow=startrow, nrow=nbatch, blc=(chanstart, -1), trc=(chanstop, -1)
            ), (3, 2, 1))
            data = PermutedDimsArray(PyCall.pycall(
                mset.tbl.getcolslice, PyCall.PyArray, datacol,
                startrow=startrow, nrow=nbatch, blc=(chanstart, -1), trc=(chanstop, -1)
            ), (3, 2, 1))

            _msetread(ch, mset.lambdas, uvw, flagrow, weight, flag, weightspectrum, data, precision)
        end
    end

    return ch
end

function _msetread(ch, lambdas, uvw, flagrow, weight, flag, weightspectrum, data, _::Type{T}) where T
    uvdatum = UVDatum{T}(
        0, 0, 0, 0, 0, SMatrix{2, 2, T, 4}(0, 0, 0, 0), SMatrix{2, 2, T, 4}(0, 0, 0, 0)
    )

    for row in axes(data, 3), chan in axes(data, 2)
        uvdatum.row = row
        uvdatum.chan = chan
        uvdatum.u = uvw[1, row] / lambdas[chan]
        uvdatum.v = -uvw[2, row] / lambdas[chan]
        uvdatum.w = uvw[3, row] / lambdas[chan]

        # We use (pol, idx) pair to convert from row-major (4,) data/weight data
        # into column-major (2, 2) shape.
        for (pol, idx) in enumerate((1, 3, 2, 4))
            d = data[pol, chan, row]
            vw = (
                !flag[pol, chan, row] *
                !flagrow[row] *
                weight[pol, row] *
                weightspectrum[pol, chan, row]
            )
            if !isfinite(vw) || !isfinite(d)
                uvdatum.data[idx] = 0
                uvdatum.weights[idx] = 0
            else
                uvdatum.data[idx] = d
                uvdatum.weights[idx] = vw
            end
        end

        put!(ch, uvdatum)
    end
end

@inline function time(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.times[uvdatum.row]
end

@inline function lambda(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.lambdas[uvdatum.chan]
end

@inline function ant1(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.ants1[uvdatum.row]
end

@inline function ant2(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.ants2[uvdatum.row]
end