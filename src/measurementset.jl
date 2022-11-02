struct MeasurementSet
    tbl::Table
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
    tbl = Table(path, Tables.Old)

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
        tbl = taql("select * from \$1 where " * join(conditions, " and "), tbl)
    end

    if chanstart == 0 && chanstop == 0
        # Select entire range
        freqs = tbl.SPECTRAL_WINDOW[:CHAN_FREQ][1]
        chanstart = 1
        chanstop = length(freqs)
    else
        freqs = tbl.SPECTRAL_WINDOW[:CHAN_FREQ][chanstart:chanstop, 1]
    end

    lambdas = c ./ freqs
    ra0, dec0 = tbl.FIELD[:PHASE_DIR][1]

    # Get time information
    times = tbl[:TIME][:]
    timesteps = unique(times)
    sort!(timesteps)

    ants1 = tbl[:ANTENNA1][:]
    ants2 = tbl[:ANTENNA2][:]

    # Get MWA phase delays
    delays = tbl.MWA_TILE_POINTING[:DELAYS][1]

    return MeasurementSet(
        tbl,
        timestart,
        timestop,
        chanstart,
        chanstop,
        size(tbl, 1),
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
        buildcache(mset, rowstart, rowstop, cache)
    end

    return cache[cacheoffset], (rowstart, cacheoffset, cache)
end

function buildcache(mset::MeasurementSet, rowstart, rowstop, cache::Vector{UVDatum{T}}) where T
    # Default to using CORRECTED_DATA if present
    datacol = :CORRECTED_DATA in propertynames(mset.tbl) ? :CORRECTED_DATA :  :DATA

    rowrange = rowstart:rowstop
    chanrange = mset.chanstart:mset.chanstop

    # We provide type assertions to ensure this function is fully typed inferred
    uvw = mset.tbl[:UVW][:, rowrange]::Matrix{Float64}
    flagrow = mset.tbl[:FLAG_ROW][rowrange]::Vector{Bool}
    weight = mset.tbl[:WEIGHT][1:4, rowrange]::Matrix{Float32}
    flag = mset.tbl[:FLAG][1:4, chanrange, rowrange]::Array{Bool, 3}
    weightspectrum = mset.tbl[:WEIGHT_SPECTRUM][:, chanrange, rowrange]::Array{Float32, 3}
    data = mset.tbl[datacol][:, chanrange, rowrange]::Array{ComplexF32, 3}

    tmpdata = zero(MMatrix{2, 2, Complex{T}, 4})
    tmpweights = zero(MMatrix{2, 2, T, 4})

    for (irow, row) in enumerate(rowrange), chan in chanrange
        u = uvw[1, irow] / mset.lambdas[chan]
        v = -uvw[2, irow] / mset.lambdas[chan]
        w = uvw[3, irow] / mset.lambdas[chan]

        # We use (pol, idx) pair to convert from row-major (4,) data/weight data
        # into column-major (2, 2) shape.
        for (pol, idx) in enumerate((1, 3, 2, 4))
            d = data[pol, chan, irow]
            vw = (
                !flag[pol, chan, irow] *
                !flagrow[irow] *
                weight[pol, irow] *
                weightspectrum[pol, chan, irow]
            )
            if !isfinite(vw) || !isfinite(d)
                tmpdata[idx] = 0
                tmpweights[idx] = 0
            else
                tmpdata[idx] = d
                tmpweights[idx] = vw
            end
        end

        if mset.ignoreflagged && iszero(tmpweights)
            continue
        end

        if w >= 0
            push!(cache, UVDatum{T}(row, chan, u, v, w, tmpweights, tmpdata))
        else
            push!(cache, UVDatum{T}(row, chan, -u, -v, -w, adjoint(tmpweights), adjoint(tmpdata)))
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