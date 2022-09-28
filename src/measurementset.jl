struct MeasurementSet
    tbl::PyCall.PyObject
    timestart::Float64
    timestop::Float64
    chanstart::Int
    chanstop::Int
    nrows::Int
    nchans::Int
    freqs::Vector{Float64}
    lambdas::Vector{Float64}
    times::Vector{Float64}
    timesteps::Vector{Float64}
    # antennas::Dict{Int, Tuple{String, @NamedTuple{x::Float64, y::Float64, z::Float64}}}
    ants1::Vector{Int}
    ants2::Vector{Int}
    phasecenter::NTuple{2, Float64}  # radians
    ignoreflagged::Bool
    delays::Vector{Integer}
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
    ignoreflagged=true
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
    if ignoreflagged
        push!(conditions, "not FLAG_ROW")
    end
    if length(conditions) != 0
        tbl = PyCasaTable.taql("select * from \$1 where " * join(conditions, " and "), tables=[tbl])
    end

    freqs = tbl.SPECTRAL_WINDOW.getcellslice("CHAN_FREQ", 0, chanstart - 1, chanstop - 1)
    lambdas = c ./ freqs
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[1, :]

    # Get time information
    times = tbl.getcol("TIME")
    timesteps = unique(times)
    sort!(timesteps)

    ants1 = tbl.getcol("ANTENNA1")
    ants2 = tbl.getcol("ANTENNA2")

    # Get MWA phase delays
    delays = tbl.MWA_TILE_POINTING.getcell("DELAYS", 0)

    return MeasurementSet(
        tbl,
        timestart,
        timestop,
        chanstart,
        chanstop,
        tbl.nrows(),
        length(freqs),
        freqs,
        lambdas,
        times,
        timesteps,
        ants1,
        ants2,
        (ra0, dec0),
        ignoreflagged,
        delays,
    )
end

Base.IteratorSize(::Type{MeasurementSet}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MeasurementSet}) = Base.HasEltype()
Base.eltype(::Type{MeasurementSet}) = UVDatum{Float64}

@inline function Base.iterate(mset::MeasurementSet)
    chunksize = 1000

    return iterate(
        mset,
        (1 - chunksize, 0, UVDatum{Float64}[]),
    )
end

@inline function Base.iterate(mset::MeasurementSet, (rowstart, cacheoffset, cache))
    chunksize = 1000

    cacheoffset += 1

    # Check if the cache is exhausted.
    # This is a while loop, as it is possible that _buildcache returns an empty cache
    # if all entries are flagged.
    while (cacheoffset > length(cache))
        rowstart += chunksize
        cacheoffset = 1

        # First, check if we're all done.
        if rowstart > mset.nrows
            return nothing
        end

        # Otherwise, it's time to rebuild the cache.
        rowstop = min(rowstart + chunksize - 1, mset.nrows)

        empty!(cache)
        sizehint!(cache, chunksize * length(mset.chanstart:mset.chanstop))
        _buildcache(mset, rowstart, rowstop, cache)
    end

    return cache[cacheoffset], (rowstart, cacheoffset, cache)
end


function _buildcache(mset::MeasurementSet, startrow, rowend, cache::Vector{UVDatum{T}}) where T
    # Default to using CORRECTED_DATA if present
    datacol = "CORRECTED_DATA" in mset.tbl.colnames() ? "CORRECTED_DATA" :  "DATA"

    # Convert channel selection from 1-indexing to Python's 0-indexing
    chanstart = mset.chanstart - 1
    chanstop = mset.chanstop - 1

    nrow = length(startrow:rowend)
    startrow -= 1  # Convert from 1 indexing to Python's 0-indexing

    uvw = PermutedDimsArray(PyCall.pycall(
        mset.tbl.getcol, PyCall.PyArray, "UVW"; startrow, nrow
    ), (2, 1))
    flagrow = PyCall.pycall(
        mset.tbl.getcol, PyCall.PyArray, "FLAG_ROW"; startrow, nrow
    )
    weight = PermutedDimsArray(PyCall.pycall(
        mset.tbl.getcol, PyCall.PyArray, "WEIGHT"; startrow, nrow
    ), (2, 1))
    flag = PermutedDimsArray(PyCall.pycall(
        mset.tbl.getcolslice, PyCall.PyArray, "FLAG";
        startrow, nrow, blc=(chanstart, -1), trc=(chanstop, -1)
    ), (3, 2, 1))
    weightspectrum = PermutedDimsArray(PyCall.pycall(
        mset.tbl.getcolslice, PyCall.PyArray, "WEIGHT_SPECTRUM";
        startrow, nrow, blc=(chanstart, -1), trc=(chanstop, -1)
    ), (3, 2, 1))
    data = PermutedDimsArray(PyCall.pycall(
        mset.tbl.getcolslice, PyCall.PyArray, datacol;
        startrow, nrow, blc=(chanstart, -1), trc=(chanstop, -1)
    ), (3, 2, 1))

    _msetread(cache, startrow, mset.lambdas, uvw, flagrow, weight, flag, weightspectrum, data, T, mset.ignoreflagged)
end

function _msetread(cache, startrow, lambdas, uvw, flagrow, weight, flag, weightspectrum, data, ::Type{T}, ignoreflagged) where T
    tmpdata = zero(MMatrix{2, 2, Complex{T}, 4})
    tmpweights = zero(MMatrix{2, 2, T, 4})

    for row in axes(data, 3), chan in axes(data, 2)
        u = uvw[1, row] / lambdas[chan]
        v = -uvw[2, row] / lambdas[chan]
        w = uvw[3, row] / lambdas[chan]

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
                tmpdata[idx] = 0
                tmpweights[idx] = 0
            else
                tmpdata[idx] = d
                tmpweights[idx] = vw
            end
        end

        if ignoreflagged && iszero(tmpweights)
            continue
        end

        if w >= 0
            push!(cache, UVDatum{T}(startrow + row, chan, u, v, w, tmpweights, tmpdata))
        else
            push!(cache, UVDatum{T}(startrow + row, chan, -u, -v, -w, adjoint(tmpweights), adjoint(tmpdata)))
        end
    end
end

@inline function time(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.times[uvdatum.row] / 86400  # division by 86400 converts MJD seconds to MJD days
end

@inline function lambda(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.lambdas[uvdatum.chan]
end

@inline function freq(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.freqs[uvdatum.chan]
end

@inline function ant1(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.ants1[uvdatum.row]
end

@inline function ant2(mset::MeasurementSet, uvdatum::UVDatum)
    return mset.ants2[uvdatum.row]
end

function meanfreq(mset::MeasurementSet, uvdata)
    totalweight = 0.
    totalfreq = 0.
    for uvdatum in uvdata
        weight = sum(uvdatum.weights)
        totalweight += weight
        totalfreq += weight * freq(mset, uvdatum)
    end
    return totalfreq / totalweight
end

function meantime(mset::MeasurementSet, uvdata)
    totalweight = 0.
    totaltime = 0.
    for uvdatum in uvdata
        weight = sum(uvdatum.weights)
        totalweight += weight
        totaltime += weight * time(mset, uvdatum)
    end
    return totaltime / totalweight
end