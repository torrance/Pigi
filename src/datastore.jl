abstract type DataStore end

"""
    read(d::DataStore, datacol=nothing)

Returns an iterator of UVDatum that are stored in `d`. The default datacol is
CORRECTED_DATA if present, or else DATA.
"""
function read(d::DataStore, datacol=nothing)
    throw(ErrorException("Invalid method call; for documentation only"))
end

"""
    write(d::DataStore, uvdata, datacol=nothing)

Returns an iterator of UVDatum that are stored in `d`. The default datacol is MODEL_DATA,
which will be created if it is not present.

uvdata must be an iterable of `UVDatum` with `row` and `chan` values that match `d`.
"""
function write(d::DataStore, uvdata, datacol=nothing)
    throw(ErrorException("Invalid method call; for documentation only"))
end